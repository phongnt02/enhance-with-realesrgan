import { calculateOptimalBitrate } from './utils.js';

export class FrameProcessor {
    async getVideoFPS(videoFile) {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(videoFile);
        
        await new Promise((resolve) => {
            video.onloadedmetadata = resolve;
        });
        
        // Improve FPS detection accuracy
        const timeRanges = [];
        let lastTime = 0;
        let frameCount = 0;
        const minSamples = 30; // Increase samples for better accuracy
        
        return new Promise((resolve) => {
            video.requestVideoFrameCallback(function callback(now, metadata) {
                const delta = metadata.mediaTime - lastTime;
                if (lastTime !== 0 && delta > 0) {
                    timeRanges.push(delta);
                }
                lastTime = metadata.mediaTime;
                frameCount++;
                
                if (frameCount >= minSamples) {
                    // Remove outliers
                    const sortedDeltas = timeRanges.sort((a, b) => a - b);
                    const q1Index = Math.floor(sortedDeltas.length * 0.25);
                    const q3Index = Math.floor(sortedDeltas.length * 0.75);
                    const validDeltas = sortedDeltas.slice(q1Index, q3Index + 1);
                    
                    const averageDelta = validDeltas.reduce((a, b) => a + b) / validDeltas.length;
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
            
            // Create array of frame timestamps
            const frameTimestamps = [];
            for (let i = 0; i < totalFrames; i++) {
                frameTimestamps.push(i * frameInterval);
            }
            
            // Extract frames in batches for better memory management
            const batchSize = 10;
            for (let i = 0; i < frameTimestamps.length; i += batchSize) {
                const batch = frameTimestamps.slice(i, i + batchSize);
                
                for (const timestamp of batch) {
                    videoElement.currentTime = timestamp;
                    
                    await new Promise(resolve => {
                        videoElement.onseeked = () => {
                            // Use requestAnimationFrame for better frame timing
                            requestAnimationFrame(() => {
                                ctx.drawImage(videoElement, 0, 0);
                                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                                frames.push({
                                    data: imageData,
                                    timestamp,
                                    index: Math.round(timestamp * fps)
                                });
                                
                                if (progressCallback) {
                                    const progress = (frames.length / totalFrames) * 100;
                                    progressCallback(Math.min(progress, 100));
                                }
                                
                                resolve();
                            });
                        };
                    });
                }
                
                // Add small delay between batches to prevent memory pressure
                await new Promise(resolve => setTimeout(resolve, 10));
            }
    
            // Sort frames by timestamp to ensure correct order
            frames.sort((a, b) => a.timestamp - b.timestamp);
            
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
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            
            // Calculate optimal bitrate based on resolution
            const bitrate = calculateOptimalBitrate(firstFrame.width, firstFrame.height);
            
            // Use higher bitrate for better quality
            const stream = canvas.captureStream(fps);
            const recorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp8',
                videoBitsPerSecond: Math.min(bitrate * 1.5, 8000000) // Increase bitrate ceiling
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
    
                    while (frameIndex <= expectedFrame && frameIndex < frames.length) {
                        // Use timing information from frame extraction
                        const frame = frames[frameIndex];
                        ctx.putImageData(frame.data, 0, 0);
                        frameIndex++;
                        
                        if (progressCallback) {
                            progressCallback((frameIndex / frames.length) * 100);
                        }
                    }
    
                    if (frameIndex < frames.length) {
                        requestAnimationFrame(renderFrame);
                    } else {
                        // Add small delay before stopping to ensure all frames are processed
                        setTimeout(() => recorder.stop(), frameDuration * 2);
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