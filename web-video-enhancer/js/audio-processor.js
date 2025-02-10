import { calculateOptimalBitrate } from './utils.js';

export class AudioProcessor {
    async extractAudioFromVideo(videoFile) {
        console.log('[extractAudio] Starting audio extraction...');
        
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({
                latencyHint: 'interactive',
                sampleRate: 48000 // Use higher sample rate
            });
            
            const videoElement = document.createElement('video');
            videoElement.src = URL.createObjectURL(videoFile);

            // Wait for metadata and ensure audio tracks are loaded
            await new Promise((resolve, reject) => {
                videoElement.onloadedmetadata = () => {
                    videoElement.oncanplay = resolve;
                };
                videoElement.onerror = reject;
            });

            const source = audioContext.createMediaElementSource(videoElement);
            const destination = audioContext.createMediaStreamDestination();
            
            // Add compression to maintain consistent audio levels
            const compressor = audioContext.createDynamicsCompressor();
            compressor.threshold.value = -24;
            compressor.knee.value = 30;
            compressor.ratio.value = 12;
            compressor.attack.value = 0.003;
            compressor.release.value = 0.25;
            
            source.connect(compressor);
            compressor.connect(destination);

            const chunks = [];
            const recorder = new MediaRecorder(destination.stream, {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 128000 // Consistent bitrate
            });

            recorder.ondataavailable = e => chunks.push(e.data);

            const recordingPromise = new Promise(resolve => {
                recorder.onstop = () => {
                    const audioBlob = new Blob(chunks, { type: 'audio/webm;codecs=opus' });
                    resolve(audioBlob);
                };
            });

            // Use precise timing for recording
            const startTime = audioContext.currentTime;
            recorder.start();
            videoElement.currentTime = 0;
            await videoElement.play();
            
            await new Promise(resolve => {
                videoElement.onended = () => {
                    // Ensure we capture all audio
                    setTimeout(resolve, 100);
                };
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
            
            // Ensure video is muted to prevent audio doubling
            videoElement.muted = true;
    
            // Wait for both elements to be ready
            await Promise.all([
                new Promise(resolve => {
                    videoElement.onloadedmetadata = () => {
                        videoElement.oncanplay = resolve;
                    };
                }),
                new Promise(resolve => {
                    audioElement.onloadedmetadata = () => {
                        audioElement.oncanplay = resolve;
                    };
                })
            ]);
    
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            const ctx = canvas.getContext('2d', {
                alpha: false,
                desynchronized: true
            });
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
    
            const stream = canvas.captureStream();
            const audioContext = new AudioContext({
                latencyHint: 'interactive',
                sampleRate: 48000
            });
            
            const source = audioContext.createMediaElementSource(audioElement);
            const dest = audioContext.createMediaStreamDestination();
            
            // Add audio processing for better sync
            const delayNode = audioContext.createDelay();
            delayNode.delayTime.value = 0.01; // Small delay for sync
            
            source.connect(delayNode);
            delayNode.connect(dest);
            
            // Add all audio tracks to ensure complete audio
            stream.getAudioTracks().forEach(track => track.enabled = false);
            dest.stream.getAudioTracks().forEach(track => {
                stream.addTrack(track);
            });
    
            const bitrate = calculateOptimalBitrate(canvas.width, canvas.height);
            const recorder = new MediaRecorder(stream, {
                mimeType: 'video/webm;codecs=vp8',
                videoBitsPerSecond: Math.min(bitrate * 1.5, 8000000),
                audioBitsPerSecond: 128000
            });
    
            const chunks = [];
            recorder.ondataavailable = e => chunks.push(e.data);
    
            return new Promise((resolve, reject) => {
                recorder.start();
    
                const duration = videoElement.duration;
                let lastDrawTime = 0;
                
                const progressInterval = setInterval(() => {
                    if (progressCallback && !videoElement.ended) {
                        const progress = (videoElement.currentTime / duration) * 100;
                        progressCallback(progress);
                    }
                }, 100);
    
                // Start playback with precise timing
                Promise.all([
                    videoElement.play().catch(reject),
                    audioElement.play().catch(reject)
                ]);
    
                function drawNextFrame(timestamp) {
                    if (!videoElement.ended) {
                        // Use timing info for smooth playback
                        const timeDiff = timestamp - lastDrawTime;
                        if (timeDiff >= (1000 / 60)) { // 60fps max
                            ctx.drawImage(videoElement, 0, 0);
                            lastDrawTime = timestamp;
                        }
                        requestAnimationFrame(drawNextFrame);
                    } else {
                        clearInterval(progressInterval);
                        // Add small delay before stopping to ensure all frames are captured
                        setTimeout(() => recorder.stop(), 100);
                    }
                }
                
                requestAnimationFrame(drawNextFrame);
    
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