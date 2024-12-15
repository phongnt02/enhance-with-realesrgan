importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

let session = null;

async function initializeONNX() {
    if (session) return;
    
    const options = {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        executionMode: 'sequential',
        enableCpuMemArena: true
    };

    session = await ort.InferenceSession.create('./model/realesrgan_web.onnx', options);
}

self.onmessage = async function(e) {
    const { tensor, width, height, x, y } = e.data;
    
    await initializeONNX();
    
    try {
        const inputShape = [1, 3, height, width];
        const inputTensor = new ort.Tensor('float32', tensor, inputShape);
        const outputs = await session.run({ input: inputTensor });
        
        self.postMessage({
            tensor: outputs.output.data,
            x,
            y,
            width,
            height
        });
    } catch (error) {
        self.postMessage({ error: error.message });
    }
};