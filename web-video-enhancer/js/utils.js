export const PROGRESS_RANGES = {
    extractFrames: [0, 20],
    extractAudio: [20, 30],
    enhance: [30, 80],
    mergeFrames: [80, 90],
    mergeAudioVideo: [90, 100]
};

export function updateProgress(progressRanges, step, currentProgress, progressCallback) {
    if (!progressCallback) return;

    const [start, end] = progressRanges[step];
    const range = end - start;
    const scaledProgress = start + (currentProgress / 100) * range;
    progressCallback(Math.round(scaledProgress));
}

export function calculateOptimalBitrate(width, height) {
    const pixels = width * height;
    const baseQuality = 0.3; // Giảm từ 0.8 xuống 0.3
    let baseBitrate = Math.round(pixels * baseQuality);
    
    if (pixels <= 921600) { // HD
        baseBitrate *= 1.5; // Giảm từ 2 xuống 1.5
    } else if (pixels <= 2073600) { // Full HD
        baseBitrate *= 2; // Giảm từ 3 xuống 2
    } else if (pixels <= 8294400) { // 4K
        baseBitrate *= 3; // Giảm từ 4 xuống 3
    }
    
    return Math.min(Math.max(baseBitrate, 800000), 4000000); // Giảm giới hạn bitrate
}