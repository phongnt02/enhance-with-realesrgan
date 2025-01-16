export class EnhanceProcessor {
    constructor() {
        this.session = null;
        this.initialized = false;
        this.modelPath = './model/realesrgan_web.onnx';
        this.scale = 4;  // Giữ lại để sau này dùng
        this.deviceInfo = 'MOCK';
    }

    async initialize() {
        this.initialized = true;
        console.log('[EnhanceProcessor] Mock initialization complete');
    }

    async enhanceFrames(frames, progressCallback) {
        console.log(`[EnhanceProcessor] Starting mock enhancement of ${frames.length} frames`);
        
        const enhancedFrames = [];

        for (let i = 0; i < frames.length; i++) {
            // Chỉ copy frame gốc, không scale up
            const enhancedData = new ImageData(
                new Uint8ClampedArray(frames[i].data.data), 
                frames[i].data.width,
                frames[i].data.height
            );

            enhancedFrames.push({
                data: enhancedData,
                timestamp: frames[i].timestamp
            });

            if (progressCallback) {
                progressCallback((i + 1) / frames.length * 100);
            }

            if ((i + 1) % Math.ceil(frames.length / 10) === 0) {
                console.log(`[EnhanceProcessor] Progress: ${Math.round((i + 1) / frames.length * 100)}%`);
            }
        }

        console.log('[EnhanceProcessor] Mock enhancement completed successfully');
        return enhancedFrames;
    }
}