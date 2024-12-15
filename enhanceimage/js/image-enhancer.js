class ImageEnhancer {
    constructor() {
        this.session = null;
        this.initialized = false;
        this.modelPath = './model/realesrgan_web.onnx';

        // Settings matching the exported model
        this.tileSize = 128;  // Khớp với TILE_SIZE trong Python
        this.tileOverlap = 8; // Khớp với TILE_OVERLAP
        this.scale = 4;       // Khớp với SCALE_FACTOR
        this.maxSize = 2048 * 2048; // Tăng max size
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

            // Enhanced ONNX session configuration
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

            console.log('[ImageEnhancer] Loading model...');
            this.session = await ort.InferenceSession.create(this.modelPath, options);

            console.log('[ImageEnhancer] Model loaded successfully');
            this.initialized = true;
        } catch (error) {
            console.error('[ImageEnhancer] Initialization failed:', error);
            throw error;
        }
    }

    async preprocessImage(imageElement) {
        const { naturalWidth, naturalHeight } = imageElement;
        const tiles = [];
        
        for (let y = 0; y < naturalHeight; y += this.tileSize) {
            for (let x = 0; x < naturalWidth; x += this.tileSize) {
                const canvas = document.createElement('canvas');
                const tileWidth = Math.min(this.tileSize, naturalWidth - x);
                const tileHeight = Math.min(this.tileSize, naturalHeight - y);
                
                canvas.width = tileWidth;
                canvas.height = tileHeight;
                
                const ctx = canvas.getContext('2d', { 
                    alpha: false,
                    willReadFrequently: true 
                });
                
                ctx.drawImage(imageElement,
                    x, y,
                    tileWidth, tileHeight,
                    0, 0,
                    tileWidth, tileHeight
                );

                const imageData = ctx.getImageData(0, 0, tileWidth, tileHeight);
                const tensor = new Float32Array(3 * tileWidth * tileHeight);

                // Normalize và chuyển sang CHW format
                for (let i = 0; i < imageData.data.length / 4; i++) {
                    // Get pixel position
                    const h = Math.floor(i / tileWidth);
                    const w = i % tileWidth;
                    
                    // Chuyển RGB sang tensor CHW format với normalization
                    for (let c = 0; c < 3; c++) {
                        const pixelValue = imageData.data[i * 4 + c];
                        const tensorIdx = c * tileWidth * tileHeight + h * tileWidth + w;
                        tensor[tensorIdx] = pixelValue / 255.0;
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

        return { tiles, originalSize: { width: naturalWidth, height: naturalHeight } };
    }

    async postprocessOutput(outputs, originalSize) {
        const canvas = document.createElement('canvas');
        canvas.width = originalSize.width * this.scale;
        canvas.height = originalSize.height * this.scale;
        
        const ctx = canvas.getContext('2d', { alpha: false });
        ctx.imageSmoothingEnabled = false;

        for (const output of outputs) {
            const { tensor, x, y, width, height } = output;
            const scaledWidth = width * this.scale;
            const scaledHeight = height * this.scale;
            
            const tileCanvas = document.createElement('canvas');
            tileCanvas.width = scaledWidth;
            tileCanvas.height = scaledHeight;
            
            const tileCtx = tileCanvas.getContext('2d', { alpha: false });
            const imageData = tileCtx.createImageData(scaledWidth, scaledHeight);

            // Chuyển tensor về RGB image
            const totalPixels = scaledWidth * scaledHeight;
            for (let p = 0; p < totalPixels; p++) {
                const outIdx = p * 4;
                const h = Math.floor(p / scaledWidth);
                const w = p % scaledWidth;

                // Lấy giá trị từ tensor format CHW
                for (let c = 0; c < 3; c++) {
                    const tensorIdx = c * totalPixels + h * scaledWidth + w;
                    const value = Math.max(0, Math.min(255, Math.round(tensor[tensorIdx] * 255)));
                    imageData.data[outIdx + c] = value;
                }
                imageData.data[outIdx + 3] = 255;  // Alpha
            }

            tileCtx.putImageData(imageData, 0, 0);
            ctx.drawImage(tileCanvas, x * this.scale, y * this.scale);
        }

        return canvas.toDataURL('image/jpeg', 1.0);
    }

    async enhance(imageElement) {
        if (!this.initialized) {
            await this.initialize();
        }
    
        try {
            console.time('Total processing');
            
            // Preprocessing
            console.time('Preprocessing');
            const { tiles, originalSize } = await this.preprocessImage(imageElement);
            console.timeEnd('Preprocessing');
            console.log(`Number of tiles to process: ${tiles.length}`);
    
            // Inference
            console.time('Inference');
            const processedTiles = [];
            for (let i = 0; i < tiles.length; i++) {
                console.time(`Tile ${i+1}/${tiles.length}`);
                
                const tile = tiles[i];
                const inputShape = [1, 3, tile.height, tile.width];
                const feeds = { 
                    input: new ort.Tensor('float32', tile.tensor, inputShape)
                };
                
                const outputs = await this.session.run(feeds);
                const outputTensor = outputs.output.data;
                
                processedTiles.push({
                    tensor: outputTensor,
                    x: tile.x,
                    y: tile.y,
                    width: tile.width,
                    height: tile.height
                });
    
                console.timeEnd(`Tile ${i+1}/${tiles.length}`);
                if ((i + 1) % 10 === 0) {
                    console.log(`Processed ${i + 1}/${tiles.length} tiles`);
                }
            }
            console.timeEnd('Inference');
    
            // Postprocessing
            console.time('Postprocessing');
            const result = await this.postprocessOutput(processedTiles, originalSize);
            console.timeEnd('Postprocessing');
    
            console.timeEnd('Total processing');
            
            return result;
            
        } catch (error) {
            console.error('[ImageEnhancer] Enhancement failed:', error);
            throw error;
        }
    }
}