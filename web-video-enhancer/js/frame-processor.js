import { calculateOptimalBitrate } from './utils.js';

export class FrameProcessor {
    async getVideoFPS(videoFile) {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(videoFile);
        
        await new Promise((resolve) => {
            video.onloadedmetadata = resolve;
        });
        
        const timeRanges = [];
        let lastTime = 0;
        
        return new Promise((resolve) => {
            video.requestVideoFrameCallback(function callback(now, metadata) {
                const delta = metadata.mediaTime - lastTime;
                if (lastTime !== 0) {
                    timeRanges.push(delta);
                }
                lastTime = metadata.mediaTime;
                
                if (timeRanges.length >= 10) {
                    const averageDelta = timeRanges.reduce((a, b) => a + b) / timeRanges.length;
                    const fps = Math.round(1 / averageDelta);
                    URL.revokeObjectURL(video.src);
                    video.remove();
                    resolve(fps);
                } else {
                    video.requestVideoFrameCallback(callback);
                }
            });
            video.play();
        });
    }

    async extractFrames(videoFile, progressCallback) {
        console.log('[extractFrames] Starting frame extraction...');
        const frames = [];
        let videoElement = null;
        
        try {
            videoElement = document.createElement('video');
            videoElement.src = URL.createObjectURL(videoFile);
            
            await new Promise((resolve, reject) => {
                videoElement.onloadedmetadata = resolve;
                videoElement.onerror = reject;
            });
    
            console.log(`[extractFrames] Video metadata loaded:`, {
                duration: videoElement.duration,
                width: videoElement.videoWidth,
                height: videoElement.videoHeight
            });
    
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d', {
                willReadFrequently: true,
                alpha: false,
                desynchronized: true
            });
            
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
    
            const duration = videoElement.duration;
            const fps = await this.getVideoFPS(videoFile);
            const frameInterval = 1 / fps;
            const totalFrames = Math.ceil(duration * fps);
    
            for (let i = 0; i < totalFrames; i++) {
                const currentTime = i * frameInterval;
                videoElement.currentTime = currentTime;
                
                await new Promise(resolve => {
                    videoElement.onseeked = () => {
                        ctx.drawImage(videoElement, 0, 0);
                        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                        frames.push({
                            data: imageData,
                            timestamp: currentTime
                        });
                        
                        if (progressCallback) {
                            const progress = (frames.length / totalFrames) * 100;
                            progressCallback(Math.min(progress, 100));
                        }
                        
                        resolve();
                    };
                });
            }
    
            console.log(`[extractFrames] Successfully extracted ${frames.length} frames`);
            return frames;
    
        } catch (error) {
            console.error('[extractFrames] Error:', error);
            throw error;
        } finally {
            if (videoElement) {
                videoElement.pause();
                URL.revokeObjectURL(videoElement.src);
                videoElement.remove();
            }
        }
    }

    async mergeFramesToVideo(frames, fps, progressCallback) {
        console.log('[mergeFrames] Starting frame merging...');
        
        try {
            const firstFrame = frames[0].data;
            const canvas = document.createElement('canvas');
            canvas.width = firstFrame.width;
            canvas.height = firstFrame.height;
            
            const ctx = canvas.getContext('2d', {
                alpha: false,
                desynchronized: true
            });
            
            const bitrate = calculateOptimalBitrate(firstFrame.width, firstFrame.height);
            const stream = canvas.captureStream(fps);
            
            const recorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp8',
                videoBitsPerSecond: Math.min(bitrate, 2000000)
            });
    
            const chunks = [];
            recorder.ondataavailable = e => chunks.push(e.data);
    
            return new Promise((resolve, reject) => {
                recorder.start();
    
                let frameIndex = 0;
                const frameDuration = 1000 / fps;
                const startTime = performance.now();
    
                function renderFrame() {
                    const currentTime = performance.now() - startTime;
                    const expectedFrame = Math.floor(currentTime / frameDuration);
    
                    if (frameIndex <= expectedFrame && frameIndex < frames.length) {
                        ctx.putImageData(frames[frameIndex].data, 0, 0);
                        frameIndex++;
                        
                        if (progressCallback) {
                            progressCallback((frameIndex / frames.length) * 100);
                        }
                    }
    
                    if (frameIndex < frames.length) {
                        requestAnimationFrame(renderFrame);
                    } else {
                        setTimeout(() => recorder.stop(), 100);
                    }
                }
    
                recorder.onstop = () => {
                    const videoBlob = new Blob(chunks, { type: 'video/webm' });
                    resolve(videoBlob);
                };
    
                requestAnimationFrame(renderFrame);
            });
            
        } catch (error) {
            console.error('[mergeFrames] Error:', error);
            throw error;
        }
    }
}