import { calculateOptimalBitrate } from './utils.js';
export class AudioProcessor {
    async extractAudioFromVideo(videoFile) {
        console.log('[extractAudio] Starting audio extraction...');
        
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const videoElement = document.createElement('video');
            videoElement.src = URL.createObjectURL(videoFile);

            await new Promise((resolve, reject) => {
                videoElement.onloadedmetadata = () => {
                    console.log('[extractAudio] Video metadata loaded, duration:', videoElement.duration);
                    resolve();
                };
                videoElement.onerror = reject;
            });

            const source = audioContext.createMediaElementSource(videoElement);
            const destination = audioContext.createMediaStreamDestination();
            source.connect(destination);

            const chunks = [];
            const recorder = new MediaRecorder(destination.stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            recorder.ondataavailable = e => chunks.push(e.data);

            const recordingPromise = new Promise(resolve => {
                recorder.onstop = () => {
                    const audioBlob = new Blob(chunks, { type: 'audio/webm;codecs=opus' });
                    console.log('[extractAudio] Audio extraction complete, size:', audioBlob.size);
                    resolve(audioBlob);
                };
            });

            recorder.start();
            await videoElement.play();
            
            await new Promise(resolve => {
                videoElement.onended = resolve;
            });

            recorder.stop();
            const audioBlob = await recordingPromise;

            // Cleanup
            videoElement.pause();
            URL.revokeObjectURL(videoElement.src);
            await audioContext.close();

            return audioBlob;

        } catch (error) {
            console.error('[extractAudio] Error:', error);
            throw error;
        }
    }

    async mergeAudioVideo(videoBlob, audioBlob, progressCallback) {
        console.log('[mergeAudioVideo] Starting audio/video merge...');
        
        try {
            const canvas = document.createElement('canvas');
            const videoElement = document.createElement('video');
            const audioElement = document.createElement('audio');
    
            videoElement.src = URL.createObjectURL(videoBlob);
            audioElement.src = URL.createObjectURL(audioBlob);
            videoElement.muted = true;
    
            await new Promise((resolve, reject) => {
                videoElement.onloadeddata = resolve;
                videoElement.onerror = reject;
            });
    
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const ctx = canvas.getContext('2d', {
                alpha: false,
                colorSpace: 'srgb'
            });
            ctx.imageSmoothingEnabled = false;
    
            const stream = canvas.captureStream();
            const audioContext = new AudioContext();
            const source = audioContext.createMediaElementSource(audioElement);
            const dest = audioContext.createMediaStreamDestination();
            source.connect(dest);
            stream.addTrack(dest.stream.getAudioTracks()[0]);
    
            const bitrate = calculateOptimalBitrate(canvas.width, canvas.height);
            
            const recorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp8',
                videoBitsPerSecond: bitrate
            });
    
            const chunks = [];
            recorder.ondataavailable = e => chunks.push(e.data);
    
            return new Promise((resolve, reject) => {
                recorder.start();
    
                const duration = videoElement.duration;
                const progressInterval = setInterval(() => {
                    if (progressCallback && !videoElement.ended) {
                        const progress = (videoElement.currentTime / duration) * 100;
                        progressCallback(progress);
                    }
                }, 100);
    
                videoElement.play().catch(reject);
                audioElement.play().catch(reject);
    
                let animationFrame;
                function drawNextFrame() {
                    if (!videoElement.ended) {
                        ctx.drawImage(videoElement, 0, 0);
                        animationFrame = requestAnimationFrame(drawNextFrame);
                    } else {
                        cancelAnimationFrame(animationFrame);
                        clearInterval(progressInterval);
                        recorder.stop();
                    }
                }
                
                drawNextFrame();
    
                recorder.onstop = () => {
                    const finalBlob = new Blob(chunks, { type: 'video/webm' });
                    cleanup();
                    resolve(finalBlob);
                };
    
                function cleanup() {
                    URL.revokeObjectURL(videoElement.src);
                    URL.revokeObjectURL(audioElement.src);
                    audioContext.close();
                    clearInterval(progressInterval);
                }
            });
    
        } catch (error) {
            console.error('[mergeAudioVideo] Error:', error);
            throw error;
        }
    }
}