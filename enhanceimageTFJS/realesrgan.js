class RealESRGAN {
    constructor() {
        this.model = null;
        this.initialized = false;
        this.TILE_SIZE = 128;
        this.SCALE_FACTOR = 4;
        this.INPUT_NODE = "serving_default_input_tensor";
        this.OUTPUT_NODE = "StatefulPartitionedCall_output_tensor";
    }

    setProgressCallback(callback) {
        this.progressCallback = callback;
    }

    updateProgress(message) {
        if (this.progressCallback) {
            this.progressCallback(message);
        }
    }

    async initialize() {
        try {
            if (tf.getBackend() !== 'webgpu') {
                await tf.setBackend('webgpu');
            }
            await tf.ready();
            console.log('Using backend:', tf.getBackend());
            
            this.model = await tf.loadGraphModel('web_model/model.json');
            console.log('Model loaded');

            // Warmup
            const warmupInput = tf.zeros([1, 64, 64, 3]);
            
            try {
                console.log('Running warmup...');
                await this.model.predict(warmupInput).array();
                console.log('Warmup complete');
            } catch (error) {
                console.error('Warmup failed:', error);
            } finally {
                warmupInput.dispose();
            }
            
            this.initialized = true;
            return true;
        } catch (error) {
            console.error('Initialization failed:', error);
            throw error;
        }
    }

    async preprocessImage(imageData) {
        return tf.tidy(() => {
            // Convert to tensor and normalize to [0,1]
            let tensor = tf.browser.fromPixels(imageData)
                .cast('float32')
                .div(255.0);
            return tensor.expandDims(0);
        });
    }

    async splitIntoTiles(tensor) {
        const [_, height, width] = tensor.shape;
        const tiles = [];
        const tilePositions = [];

        const numTilesY = Math.ceil(height / this.TILE_SIZE);
        const numTilesX = Math.ceil(width / this.TILE_SIZE);

        for (let y = 0; y < numTilesY; y++) {
            for (let x = 0; x < numTilesX; x++) {
                const tileStartY = y * this.TILE_SIZE;
                const tileStartX = x * this.TILE_SIZE;
                const tileHeight = Math.min(this.TILE_SIZE, height - tileStartY);
                const tileWidth = Math.min(this.TILE_SIZE, width - tileStartX);

                const tile = tf.tidy(() => {
                    let tileTensor = tensor.slice(
                        [0, tileStartY, tileStartX, 0],
                        [1, tileHeight, tileWidth, 3]
                    );

                    if (tileHeight < this.TILE_SIZE || tileWidth < this.TILE_SIZE) {
                        tileTensor = tf.pad(tileTensor, [
                            [0, 0],
                            [0, this.TILE_SIZE - tileHeight],
                            [0, this.TILE_SIZE - tileWidth],
                            [0, 0]
                        ]);
                    }

                    return tileTensor;
                });

                tiles.push(tile);
                tilePositions.push({
                    x: tileStartX,
                    y: tileStartY,
                    width: tileWidth,
                    height: tileHeight
                });
            }
        }

        return { tiles, tilePositions };
    }

    async upscaleTile(tile) {
        return tf.tidy(() => {
            const result = this.model.predict(tile);
            return result;
        });
    }

    async mergeTiles(upscaledTiles, tilePositions, originalShape) {
        const [_, height, width] = originalShape;
        const upscaledHeight = height * this.SCALE_FACTOR;
        const upscaledWidth = width * this.SCALE_FACTOR;

        return tf.tidy(() => {
            const output = tf.buffer([1, upscaledHeight, upscaledWidth, 3]);

            for (let i = 0; i < upscaledTiles.length; i++) {
                const tile = upscaledTiles[i];
                const pos = tilePositions[i];
                
                const scaledPos = {
                    x: pos.x * this.SCALE_FACTOR,
                    y: pos.y * this.SCALE_FACTOR,
                    width: pos.width * this.SCALE_FACTOR,
                    height: pos.height * this.SCALE_FACTOR
                };

                const tileData = tile.arraySync()[0];
                
                for (let y = 0; y < scaledPos.height; y++) {
                    for (let x = 0; x < scaledPos.width; x++) {
                        for (let c = 0; c < 3; c++) {
                            output.set(tileData[y][x][c], 0, scaledPos.y + y, scaledPos.x + x, c);
                        }
                    }
                }
            }

            return output.toTensor();
        });
    }

    async upscale(imageElement) {
        if (!this.initialized) {
            throw new Error('Model not initialized. Call initialize() first.');
        }

        let inputTensor = null;
        let tiles = [];
        let upscaledTiles = [];
        let mergedResult = null;

        try {
            this.updateProgress('Preprocessing image...');
            inputTensor = await this.preprocessImage(imageElement);
            console.log('Input shape:', inputTensor.shape);

            this.updateProgress('Splitting into tiles...');
            const { tiles: imageTiles, tilePositions } = await this.splitIntoTiles(inputTensor);
            tiles = imageTiles;
            console.log('Split into', tiles.length, 'tiles');

            this.updateProgress('Processing tiles...');
            upscaledTiles = [];
            for (let i = 0; i < tiles.length; i++) {
                this.updateProgress(`Processing tile ${i + 1}/${tiles.length}...`);
                console.log('Processing tile', i, 'shape:', tiles[i].shape);
                const upscaledTile = await this.upscaleTile(tiles[i]);
                console.log('Upscaled tile shape:', upscaledTile.shape);
                upscaledTiles.push(upscaledTile);
            }

            this.updateProgress('Merging tiles...');
            mergedResult = await this.mergeTiles(upscaledTiles, tilePositions, inputTensor.shape);
            console.log('Merged result shape:', mergedResult.shape);

            this.updateProgress('Converting to image...');
            // Scale back to [0,255] range
            const finalResult = tf.tidy(() => {
                return tf.clipByValue(mergedResult.mul(255.0), 0, 255);
            });
            const outputArray = await tf.browser.toPixels(finalResult);
            finalResult.dispose();
            
            return new ImageData(
                outputArray,
                inputTensor.shape[2] * this.SCALE_FACTOR,
                inputTensor.shape[1] * this.SCALE_FACTOR
            );

        } catch (error) {
            console.error('Upscale failed:', error);
            throw error;
        } finally {
            // Cleanup
            if (inputTensor) inputTensor.dispose();
            tiles.forEach(t => t?.dispose());
            upscaledTiles.forEach(t => t?.dispose());
            if (mergedResult) mergedResult.dispose();
        }
    }
}