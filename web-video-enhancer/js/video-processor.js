import { PROGRESS_RANGES, updateProgress } from './utils.js';
import { FrameProcessor } from './frame-processor.js';
import { AudioProcessor } from './audio-processor.js';
import { EnhanceProcessor } from './enhance-processor.js';
import { VideoMetrics } from './metrics.js';

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
            const startTime = performance.now();
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
            
            // Calculate and log additional metrics
            const endTime = performance.now();
            const processingTime = (endTime - startTime) / 1000;
            const qualityMetrics = VideoMetrics.calculateQualityMetrics(frames, enhancedFrames);

            console.log('\n=== Additional Processing Metrics ===');
            console.log(`Total processing time: ${processingTime.toFixed(2)} seconds`);
            console.log(`Processing device: ${this.enhanceProcessor.deviceInfo || 'WASM (Default)'}`);
            console.log(`Original dimensions: ${qualityMetrics.originalSize.width}x${qualityMetrics.originalSize.height}`);
            console.log(`Enhanced dimensions: ${qualityMetrics.enhancedSize.width}x${qualityMetrics.enhancedSize.height}`);
            console.log(`Quality Metrics:`);
            console.log('Original Video (Baseline):');
            console.log(`- PSNR: ${qualityMetrics.baselinePSNR.toFixed(2)} dB`);
            console.log(`- SSIM: ${qualityMetrics.baselineSSIM.toFixed(4)}`);
            console.log('Enhanced Video:');
            console.log(`- PSNR: ${qualityMetrics.enhancedPSNR.toFixed(2)} dB`);
            console.log(`- SSIM: ${qualityMetrics.enhancedSSIM.toFixed(4)}`);
            console.log(`Quality Improvement:`);
            console.log(`- PSNR: ${(qualityMetrics.enhancedPSNR - qualityMetrics.baselinePSNR).toFixed(2)} dB`);
            console.log(`- SSIM: ${(qualityMetrics.enhancedSSIM - qualityMetrics.baselineSSIM).toFixed(4)}`);
            console.log('=== End of Additional Metrics ===\n');

            return finalVideo;

        } catch (error) {
            console.error('Test failed:', error);
            throw error;
        }
    }
}