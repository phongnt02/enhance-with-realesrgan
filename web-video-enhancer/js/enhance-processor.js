export class EnhanceProcessor {
    constructor() {
        this.session = null;
        this.initialized = false;
        this.modelPath = './model/realesrgan_web.onnx';
        
        // Settings phải match với model requirements
        this.tileSize = 256;  // Phải giữ 256 để match với model input size
        this.tileOverlap = 8; // Giữ nguyên overlap
        this.scale = 4;       // Scale factor
        this.maxSize = 2048 * 2048; // Tăng max size
        
        this.batchSize = 2;
        this.maxConcurrent = 2;
        
        this.deviceInfo = this.getDeviceInfo();
        this.previousTimestamp = null;
    }

    getDeviceInfo() {
        const gpu = this.getGPUInfo();
        return `CPU: ${navigator.hardwareConcurrency} cores${gpu ? `, GPU: ${gpu}` : ''}`;
    }

    getGPUInfo() {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (!gl) return null;
        
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        if (!debugInfo) return 'WebGL Support';
        
        return gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    }

    async waitForOrt() {
        let attempts = 0;
        const maxAttempts = 100;

        while (attempts < maxAttempts) {
            if (window.ort) {
                return true;
            }
            await new Promise(resolve => setTimeout(resolve, 100));
            attempts++;
        }
        throw new Error('ONNX Runtime failed to load after 10 seconds');
    }

    async initialize() {
        if (this.initialized) return;

        try {
            await this.waitForOrt();
            
            const hasWebGL = () => {
                try {
                    const canvas = document.createElement('canvas');
                    return !!(window.WebGL2RenderingContext && 
                             canvas.getContext('webgl2')) || 
                           !!(window.WebGLRenderingContext && 
                             canvas.getContext('webgl'));
                } catch (e) {
                    return false;
                }
            };

            // Cấu hình tối ưu như image enhancer
            const options = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                executionMode: 'sequential',
                enableCpuMemArena: true,
                extra: {
                    session: {
                        intra_op_num_threads: navigator.hardwareConcurrency || 4,
                        inter_op_num_threads: navigator.hardwareConcurrency || 4,
                        optimization_level: 3
                    }
                }
            };

            if (hasWebGL()) {
                options.extra.session = {
                    ...options.extra.session,
                    use_webgl: true,
                    webgl_pack_unpack_optimizations: true
                };
                console.log('[EnhanceProcessor] WebGL optimizations enabled');
            }

            console.log('[EnhanceProcessor] Loading model...');
            this.session = await ort.InferenceSession.create(this.modelPath, options);
            
            console.log('[EnhanceProcessor] Model loaded successfully');
            this.initialized = true;
        } catch (error) {
            console.error('[EnhanceProcessor] Initialization failed:', error);
            throw error;
        }
    }

    async preprocessFrame(frame) {
        const { width, height } = frame.data;
        const tiles = [];
        
        // Validate dimensions
        console.log(`[EnhanceProcessor] Processing frame: ${width}x${height}`);
        if (width * height > this.maxSize) {
            throw new Error(`Frame size ${width}x${height} exceeds maximum allowed size`);
        }

        // Create canvas for the frame
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d', { 
            alpha: false, 
            willReadFrequently: true 
        });
        ctx.putImageData(frame.data, 0, 0);

        // Tối ưu lại logic xử lý tile như image enhancer
        for (let y = 0; y < height; y += this.tileSize) {
            for (let x = 0; x < width; x += this.tileSize) {
                const tileWidth = Math.min(this.tileSize, width - x);
                const tileHeight = Math.min(this.tileSize, height - y);
                
                const tileCanvas = document.createElement('canvas');
                tileCanvas.width = tileWidth;
                tileCanvas.height = tileHeight;
                
                const tileCtx = tileCanvas.getContext('2d', { 
                    alpha: false,
                    willReadFrequently: true 
                });

                tileCtx.drawImage(canvas,
                    x, y, tileWidth, tileHeight,
                    0, 0, tileWidth, tileHeight
                );

                const imageData = tileCtx.getImageData(0, 0, tileWidth, tileHeight);
                // Create tensor với fixed size 256x256
                const tensor = new Float32Array(3 * this.tileSize * this.tileSize);

                // Normalize và chuyển sang CHW format với padding
                for (let i = 0; i < imageData.data.length / 4; i++) {
                    const h = Math.floor(i / tileWidth);
                    const w = i % tileWidth;
                    
                    // Chuyển RGB sang tensor CHW format với normalization
                    for (let c = 0; c < 3; c++) {
                        const pixelValue = imageData.data[i * 4 + c];
                        // Map vào tensor 256x256
                        const tensorIdx = c * this.tileSize * this.tileSize + h * this.tileSize + w;
                        tensor[tensorIdx] = pixelValue / 255.0;
                    }
                }

                // Fill padding areas với edge values
                for (let c = 0; c < 3; c++) {
                    // Pad width
                    for (let h = 0; h < tileHeight; h++) {
                        for (let w = tileWidth; w < this.tileSize; w++) {
                            const edgeIdx = c * this.tileSize * this.tileSize + h * this.tileSize + (tileWidth - 1);
                            const padIdx = c * this.tileSize * this.tileSize + h * this.tileSize + w;
                            tensor[padIdx] = tensor[edgeIdx];
                        }
                    }
                    // Pad height
                    for (let h = tileHeight; h < this.tileSize; h++) {
                        for (let w = 0; w < this.tileSize; w++) {
                            const edgeIdx = c * this.tileSize * this.tileSize + (tileHeight - 1) * this.tileSize + w;
                            const padIdx = c * this.tileSize * this.tileSize + h * this.tileSize + w;
                            tensor[padIdx] = tensor[edgeIdx];
                        }
                    }
                }

                tiles.push({
                    tensor,
                    x,
                    y,
                    width: tileWidth,
                    height: tileHeight
                });
            }
        }

        return { tiles, originalSize: { width, height } };
    }

    async postprocessTiles(processedTiles, originalSize) {
        const canvas = document.createElement('canvas');
        canvas.width = originalSize.width * this.scale;
        canvas.height = originalSize.height * this.scale;
        
        const ctx = canvas.getContext('2d', { alpha: false });
        ctx.imageSmoothingEnabled = false;

        for (const output of processedTiles) {
            const { tensor, x, y, width, height } = output;
            const scaledWidth = width * this.scale;
            const scaledHeight = height * this.scale;
            
            const tileCanvas = document.createElement('canvas');
            tileCanvas.width = scaledWidth;
            tileCanvas.height = scaledHeight;
            
            const tileCtx = tileCanvas.getContext('2d', { alpha: false });
            const imageData = tileCtx.createImageData(scaledWidth, scaledHeight);

            // Tối ưu lại logic chuyển tensor về RGB như image enhancer
            // Lấy vùng data thực từ tensor đã được enhance
            const scaledTileSize = this.tileSize * this.scale; // 256 * 4 = 1024
            for (let h = 0; h < scaledHeight; h++) {
                for (let w = 0; w < scaledWidth; w++) {
                    const outIdx = (h * scaledWidth + w) * 4;
                    
                    for (let c = 0; c < 3; c++) {
                        const tensorIdx = c * scaledTileSize * scaledTileSize + h * scaledTileSize + w;
                        const value = Math.max(0, Math.min(255, Math.round(tensor[tensorIdx] * 255)));
                        imageData.data[outIdx + c] = value;
                    }
                    imageData.data[outIdx + 3] = 255;
                }
            }

            tileCtx.putImageData(imageData, 0, 0);
            ctx.drawImage(tileCanvas, x * this.scale, y * this.scale);
        }

        return ctx.getImageData(0, 0, canvas.width, canvas.height);
    }

    async enhanceFrame(frame, frameIndex) {
        try {
            const startTime = performance.now();
            
            // Đảm bảo timing nhất quán giữa các frame
            if (this.previousTimestamp && frame.timestamp) {
                const expectedDelta = frame.timestamp - this.previousTimestamp;
                const actualDelta = startTime - this.lastProcessTime;
                if (actualDelta < expectedDelta) {
                    await new Promise(resolve => setTimeout(resolve, expectedDelta - actualDelta));
                }
            }
            
            const { tiles, originalSize } = await this.preprocessFrame(frame);
            const processedTiles = [];
            
            for (let i = 0; i < tiles.length; i++) {
                const tile = tiles[i];
                const inputShape = [1, 3, this.tileSize, this.tileSize];
                const feeds = { 
                    input: new ort.Tensor('float32', tile.tensor, inputShape)
                };

                const outputs = await this.session.run(feeds);
                processedTiles.push({
                    tensor: outputs.output.data,
                    x: tile.x,
                    y: tile.y,
                    width: tile.width,
                    height: tile.height
                });
            }

            // Process các tile theo thứ tự không gian để tránh artifacts
            processedTiles.sort((a, b) => {
                if (a.y === b.y) return a.x - b.x;
                return a.y - b.y;
            });

            const enhancedImageData = await this.postprocessTiles(processedTiles, originalSize);
            
            // Cập nhật tracking timing
            this.previousTimestamp = frame.timestamp;
            this.lastProcessTime = performance.now();
            
            return {
                data: enhancedImageData,
                timestamp: frame.timestamp,
                originalTimestamp: frame.timestamp // Lưu lại timestamp gốc
            };

        } catch (error) {
            console.error(`[EnhanceProcessor] Frame ${frameIndex} enhancement failed:`, error);
            throw error;
        }
    }

    async enhanceFrames(frames, progressCallback) {
        if (!this.initialized) {
            await this.initialize();
        }

        const totalFrames = frames.length;
        const enhancedFrames = [];
        
        // Tính toán frame interval từ timestamps
        const frameIntervals = [];
        for (let i = 1; i < frames.length; i++) {
            frameIntervals.push(frames[i].timestamp - frames[i-1].timestamp);
        }
        const averageInterval = frameIntervals.reduce((a, b) => a + b, 0) / frameIntervals.length;
        
        console.log(`[EnhanceProcessor] Average frame interval: ${averageInterval.toFixed(2)}ms`);

        // Reset timing tracking
        this.previousTimestamp = null;
        this.lastProcessTime = null;

        for (let i = 0; i < frames.length; i += this.batchSize) {
            const batchStartTime = performance.now();
            const batch = frames.slice(i, Math.min(i + this.batchSize, frames.length));
            
            try {
                // Xử lý frame theo batch với timing control
                const enhancedBatch = await Promise.all(
                    batch.map(async (frame, idx) => {
                        // Đảm bảo timing nhất quán trong batch
                        if (idx > 0) {
                            const targetDelay = averageInterval * idx;
                            const currentDelay = performance.now() - batchStartTime;
                            if (currentDelay < targetDelay) {
                                await new Promise(resolve => setTimeout(resolve, targetDelay - currentDelay));
                            }
                        }
                        return this.enhanceFrame(frame, i + idx + 1);
                    })
                );

                enhancedFrames.push(...enhancedBatch);

                if (progressCallback) {
                    const progress = ((i + batch.length) / totalFrames * 100);
                    progressCallback(progress);
                }

                // Đảm bảo timing giữa các batch
                const batchTime = performance.now() - batchStartTime;
                const targetBatchTime = averageInterval * batch.length;
                if (batchTime < targetBatchTime) {
                    await new Promise(resolve => setTimeout(resolve, targetBatchTime - batchTime));
                }

            } catch (error) {
                console.error(`[EnhanceProcessor] Error processing batch ${Math.floor(i/this.batchSize) + 1}:`, error);
                throw error;
            }
        }

        // Sort theo timestamp gốc để đảm bảo thứ tự đúng
        enhancedFrames.sort((a, b) => a.originalTimestamp - b.originalTimestamp);

        console.log('[EnhanceProcessor] Enhancement completed successfully');
        return enhancedFrames;
    }
}