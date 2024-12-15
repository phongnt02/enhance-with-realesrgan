export class EnhanceProcessor {
    constructor() {
        this.session = null;
        this.initialized = false;
        this.modelPath = './model/realesrgan_web.onnx';
        this.inputShape = [1, 3, 256, 256];
    }

    async initialize() {
        if (this.initialized) return;

        try {
            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                enableMemPattern: true,
                executionMode: 'sequential'
            });
            
            console.log('[EnhanceProcessor] Model loaded successfully');
            console.log('[EnhanceProcessor] Input shape:', this.inputShape);
            
            this.initialized = true;
        } catch (error) {
            console.error('[EnhanceProcessor] Model initialization failed:', error);
            throw new Error('Failed to initialize EnhanceProcessor: ' + error.message);
        }
    }

    async preprocessFrame(imageData) {
        // Tạo canvas với chất lượng cao
        const originalCanvas = document.createElement('canvas');
        originalCanvas.width = imageData.width;
        originalCanvas.height = imageData.height;
        const originalCtx = originalCanvas.getContext('2d', { 
            alpha: false,
            willReadFrequently: true
        });
        originalCtx.imageSmoothingEnabled = true;
        originalCtx.imageSmoothingQuality = 'high';
        originalCtx.putImageData(imageData, 0, 0);

        // Resize với chất lượng cao
        const resizeCanvas = document.createElement('canvas');
        resizeCanvas.width = this.inputShape[2];
        resizeCanvas.height = this.inputShape[3];
        const resizeCtx = resizeCanvas.getContext('2d', {
            alpha: false,
            willReadFrequently: true
        });
        resizeCtx.imageSmoothingEnabled = true;
        resizeCtx.imageSmoothingQuality = 'high';

        // Áp dụng resize với preserved aspect ratio
        const scale = Math.min(
            this.inputShape[2] / imageData.width,
            this.inputShape[3] / imageData.height
        );
        const scaledWidth = imageData.width * scale;
        const scaledHeight = imageData.height * scale;
        const offsetX = (this.inputShape[2] - scaledWidth) / 2;
        const offsetY = (this.inputShape[3] - scaledHeight) / 2;

        // Fill background
        resizeCtx.fillStyle = '#000000';
        resizeCtx.fillRect(0, 0, this.inputShape[2], this.inputShape[3]);
        
        // Draw image
        resizeCtx.drawImage(
            originalCanvas,
            offsetX, offsetY,
            scaledWidth, scaledHeight
        );

        const resizedImageData = resizeCtx.getImageData(
            0, 0,
            this.inputShape[2],
            this.inputShape[3]
        );
        
        // Convert to tensor với format BGR (theo yêu cầu của model)
        const inputTensor = new Float32Array(this.inputShape.reduce((a, b) => a * b));
        const size = this.inputShape[2] * this.inputShape[3];
        
        for (let i = 0; i < size; i++) {
            const pixelIndex = i * 4;
            
            // Chuyển đổi RGB -> BGR và normalize
            inputTensor[i] = resizedImageData.data[pixelIndex + 2] / 255.0;                    // B
            inputTensor[i + size] = resizedImageData.data[pixelIndex + 1] / 255.0;            // G
            inputTensor[i + size * 2] = resizedImageData.data[pixelIndex] / 255.0;            // R
        }

        return inputTensor;
    }

    async postprocessFrame(outputTensor, originalWidth, originalHeight) {
        const upscaledSize = this.inputShape[2] * 4;
        
        // Tạo canvas với chất lượng cao
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = upscaledSize;
        tempCanvas.height = upscaledSize;
        const ctx = tempCanvas.getContext('2d', {
            alpha: false,
            willReadFrequently: true
        });
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        const outputImageData = ctx.createImageData(upscaledSize, upscaledSize);
        const size = upscaledSize * upscaledSize;

        // Chuyển đổi BGR -> RGB và denormalize
        for (let i = 0; i < size; i++) {
            const pixelIndex = i * 4;
            
            // Chuyển đổi BGR -> RGB
            outputImageData.data[pixelIndex] = Math.min(255, Math.max(0, outputTensor[i + size * 2] * 255));     // R
            outputImageData.data[pixelIndex + 1] = Math.min(255, Math.max(0, outputTensor[i + size] * 255));     // G
            outputImageData.data[pixelIndex + 2] = Math.min(255, Math.max(0, outputTensor[i] * 255));            // B
            outputImageData.data[pixelIndex + 3] = 255;
        }

        ctx.putImageData(outputImageData, 0, 0);

        // Calculate scaled dimensions maintaining aspect ratio
        const scale = Math.min(
            (originalWidth * 4) / upscaledSize,
            (originalHeight * 4) / upscaledSize
        );
        const finalWidth = originalWidth * 4;
        const finalHeight = originalHeight * 4;

        // Create final output với chất lượng cao
        const finalCanvas = document.createElement('canvas');
        finalCanvas.width = finalWidth;
        finalCanvas.height = finalHeight;
        const finalCtx = finalCanvas.getContext('2d', {
            alpha: false,
            willReadFrequently: true
        });
        finalCtx.imageSmoothingEnabled = true;
        finalCtx.imageSmoothingQuality = 'high';

        // Draw với preserved aspect ratio
        finalCtx.drawImage(
            tempCanvas,
            0, 0,
            finalWidth, finalHeight
        );
        
        return finalCtx.getImageData(0, 0, finalWidth, finalHeight);
    }

    async enhanceFrames(frames, progressCallback) {
        if (!this.initialized) {
            await this.initialize();
        }

        const enhancedFrames = [];
        const totalFrames = frames.length;
        console.log(`[EnhanceProcessor] Starting enhancement of ${totalFrames} frames`);

        for (let i = 0; i < totalFrames; i++) {
            try {
                const frame = frames[i];
                const inputTensor = await this.preprocessFrame(frame.data);
                
                const feeds = { 
                    input: new ort.Tensor('float32', inputTensor, this.inputShape)
                };
                const outputMap = await this.session.run(feeds);
                const outputTensor = outputMap.output.data;

                const enhancedImageData = await this.postprocessFrame(
                    outputTensor,
                    frame.data.width,
                    frame.data.height
                );

                enhancedFrames.push({
                    data: enhancedImageData,
                    timestamp: frame.timestamp
                });

                if (progressCallback) {
                    progressCallback((i + 1) / totalFrames * 100);
                }

                if ((i + 1) % Math.ceil(totalFrames / 10) === 0) {
                    console.log(`[EnhanceProcessor] Progress: ${Math.round((i + 1) / totalFrames * 100)}%`);
                }

            } catch (error) {
                console.error(`[EnhanceProcessor] Error processing frame ${i}:`, error);
                throw error;
            }
        }

        console.log('[EnhanceProcessor] Enhancement completed successfully');
        return enhancedFrames;
    }
}