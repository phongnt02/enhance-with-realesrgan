import { PROGRESS_RANGES, updateProgress } from './utils.js';
import { FrameProcessor } from './frame-processor.js';
import { AudioProcessor } from './audio-processor.js';
import { EnhanceProcessor } from './enhance-processor.js';

export class VideoProcessor {
    constructor() {
        // Initialize processors
        this.frameProcessor = new FrameProcessor();
        this.audioProcessor = new AudioProcessor();
        this.enhanceProcessor = new EnhanceProcessor();
        this.progressRanges = PROGRESS_RANGES;
        
        // Basic settings
        this.processing = false;
        this.currentStep = 0;
        this.progress = 0;
    }

    updateProgress(step, currentProgress, progressCallback) {
        if (!progressCallback) return;
        updateProgress(this.progressRanges, step, currentProgress, progressCallback);
    }

    async testBasicFunctionality(videoFile, progressCallback) {
        try {
            console.log('Starting basic functionality test...');
            
            // Get original FPS
            const originalFPS = await this.frameProcessor.getVideoFPS(videoFile);
            console.log('Original video FPS:', originalFPS);
            
            // Extract frames with original FPS
            const frames = await this.frameProcessor.extractFrames(videoFile, progress => {
                this.updateProgress('extractFrames', progress, progressCallback);
            });
            
            // Extract audio
            const audioBlob = await this.audioProcessor.extractAudioFromVideo(videoFile);
            this.updateProgress('extractAudio', 100, progressCallback);
            
            // Enhance frames (currently simulation)
            const enhancedFrames = await this.enhanceProcessor.enhanceFrames(frames, progress => {
                this.updateProgress('enhance', progress, progressCallback);
            });
            
            // Merge frames back to video
            const videoBlob = await this.frameProcessor.mergeFramesToVideo(enhancedFrames, originalFPS, progress => {
                this.updateProgress('mergeFrames', progress, progressCallback);
            });
            
            // Final merge with audio
            const finalVideo = await this.audioProcessor.mergeAudioVideo(videoBlob, audioBlob, progress => {
                this.updateProgress('mergeAudioVideo', progress, progressCallback);
            });
            
            console.log('Basic functionality test complete!');
            return finalVideo;
            
        } catch (error) {
            console.error('Test failed:', error);
            throw error;
        }
    }
}