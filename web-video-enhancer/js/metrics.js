export class VideoMetrics {
    // Helper function to resize ImageData to target dimensions
    static resizeImageData(imageData, targetWidth, targetHeight) {
        const canvas = document.createElement('canvas');
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        
        const ctx = canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);
        
        const resizeCanvas = document.createElement('canvas');
        resizeCanvas.width = targetWidth;
        resizeCanvas.height = targetHeight;
        
        const resizeCtx = resizeCanvas.getContext('2d');
        resizeCtx.drawImage(canvas, 0, 0, targetWidth, targetHeight);
        
        return resizeCtx.getImageData(0, 0, targetWidth, targetHeight);
    }

    // Calculate PSNR between two ImageData objects
    static calculatePSNR(originalImg, enhancedImg) {
        let img1 = originalImg;
        let img2 = enhancedImg;

        // If dimensions don't match, resize enhanced image to original size for comparison
        if (originalImg.width !== enhancedImg.width || originalImg.height !== enhancedImg.height) {
            img2 = this.resizeImageData(enhancedImg, originalImg.width, originalImg.height);
        }

        let mse = 0;
        const data1 = img1.data;
        const data2 = img2.data;

        // Calculate MSE for each channel (R,G,B)
        for (let i = 0; i < data1.length; i += 4) {
            mse += Math.pow(data1[i] - data2[i], 2); // R
            mse += Math.pow(data1[i + 1] - data2[i + 1], 2); // G
            mse += Math.pow(data1[i + 2] - data2[i + 2], 2); // B
        }

        mse = mse / (img1.width * img1.height * 3);

        if (mse === 0) return Infinity;
        const maxPixelValue = 255;
        const psnr = 20 * Math.log10(maxPixelValue) - 10 * Math.log10(mse);
        return psnr;
    }

    // Calculate SSIM between two ImageData objects
    static calculateSSIM(originalImg, enhancedImg) {
        let img1 = originalImg;
        let img2 = enhancedImg;

        // If dimensions don't match, resize enhanced image to original size for comparison
        if (originalImg.width !== enhancedImg.width || originalImg.height !== enhancedImg.height) {
            img2 = this.resizeImageData(enhancedImg, originalImg.width, originalImg.height);
        }

        const L = 255; // Dynamic range
        const k1 = 0.01;
        const k2 = 0.03;
        const c1 = Math.pow(k1 * L, 2);
        const c2 = Math.pow(k2 * L, 2);

        let sum1 = 0, sum2 = 0;
        let sum1_sq = 0, sum2_sq = 0;
        let sum_12 = 0;
        
        const data1 = img1.data;
        const data2 = img2.data;
        const numPixels = img1.width * img1.height;

        // Calculate intermediate values
        for (let i = 0; i < data1.length; i += 4) {
            // Convert to grayscale using luminance formula
            const p1 = 0.299 * data1[i] + 0.587 * data1[i + 1] + 0.114 * data1[i + 2];
            const p2 = 0.299 * data2[i] + 0.587 * data2[i + 1] + 0.114 * data2[i + 2];

            sum1 += p1;
            sum2 += p2;
            sum1_sq += p1 * p1;
            sum2_sq += p2 * p2;
            sum_12 += p1 * p2;
        }

        // Calculate means and variances
        const mu1 = sum1 / numPixels;
        const mu2 = sum2 / numPixels;
        const sigma1_sq = (sum1_sq / numPixels) - (mu1 * mu1);
        const sigma2_sq = (sum2_sq / numPixels) - (mu2 * mu2);
        const sigma12 = (sum_12 / numPixels) - (mu1 * mu2);

        // Calculate SSIM
        const ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) /
                    ((mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2));

        return ssim;
    }

    static calculateBaselineMetrics(frames) {
        let totalPSNR = 0;
        let totalSSIM = 0;
        const numFrames = frames.length;

        // So sánh mỗi frame với frame tiếp theo để tạo baseline
        for (let i = 0; i < numFrames - 1; i++) {
            const psnr = this.calculatePSNR(frames[i].data, frames[i + 1].data);
            const ssim = this.calculateSSIM(frames[i].data, frames[i + 1].data);
            
            totalPSNR += psnr;
            totalSSIM += ssim;
        }

        return {
            averagePSNR: totalPSNR / (numFrames - 1),
            averageSSIM: totalSSIM / (numFrames - 1)
        };
    }

    // Cập nhật hàm calculateQualityMetrics để bao gồm cả baseline
    static calculateQualityMetrics(originalFrames, enhancedFrames) {
        let totalPSNR = 0;
        let totalSSIM = 0;
        const numFrames = Math.min(originalFrames.length, enhancedFrames.length);
        
        console.log(`[VideoMetrics] Comparing frames - Original: ${originalFrames[0].data.width}x${originalFrames[0].data.height}, Enhanced: ${enhancedFrames[0].data.width}x${enhancedFrames[0].data.height}`);

        // Tính baseline metrics từ original frames
        const baselineMetrics = this.calculateBaselineMetrics(originalFrames);
        console.log('[VideoMetrics] Baseline metrics calculated');

        // Tính metrics cho enhanced frames
        for (let i = 0; i < numFrames; i++) {
            const psnr = this.calculatePSNR(originalFrames[i].data, enhancedFrames[i].data);
            const ssim = this.calculateSSIM(originalFrames[i].data, enhancedFrames[i].data);
            
            totalPSNR += psnr;
            totalSSIM += ssim;
        }

        return {
            // Enhanced metrics
            enhancedPSNR: totalPSNR / numFrames,
            enhancedSSIM: totalSSIM / numFrames,
            // Baseline metrics
            baselinePSNR: baselineMetrics.averagePSNR,
            baselineSSIM: baselineMetrics.averageSSIM,
            // Size info
            originalSize: {
                width: originalFrames[0].data.width,
                height: originalFrames[0].data.height
            },
            enhancedSize: {
                width: enhancedFrames[0].data.width,
                height: enhancedFrames[0].data.height
            }
        };
    }
}