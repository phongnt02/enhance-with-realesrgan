import { VideoProcessor } from './video-processor.js';
document.addEventListener('DOMContentLoaded', async function() {
    // Get DOM elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'video/*';
    fileInput.style.display = 'none';
    document.body.appendChild(fileInput);

    const progress = document.getElementById('progress');
    const status = document.getElementById('status');
    const processButton = document.querySelector('.button-container').children[0];
    const downloadButton = document.querySelector('.button-container').children[1];
    const originalVideo = document.createElement('video');
    originalVideo.controls = true;
    const enhancedVideo = document.createElement('video');
    enhancedVideo.controls = true;

    // Add videos to containers
    document.querySelectorAll('.video-box').forEach((box, index) => {
        if (index === 0) {
            box.appendChild(originalVideo);
        } else {
            box.appendChild(enhancedVideo);
        }
    });

    // Verify all elements
    const elements = {
        dropZone, fileInput, progress, status, processButton,
        downloadButton, originalVideo, enhancedVideo
    };

    for (const [name, element] of Object.entries(elements)) {
        if (!element) {
            console.error(`Element not found: ${name}`);
            throw new Error(`Required element ${name} not found`);
        }
    }

    let currentFile = null;
    const processor = new VideoProcessor();

    // File handling functions
    function handleFile(file) {
        if (!file) {
            status.textContent = 'No file selected';
            return;
        }

        if (!file.type.startsWith('video/')) {
            status.textContent = 'Please select a valid video file';
            return;
        }

        // Update UI for new file
        currentFile = file;
        originalVideo.src = URL.createObjectURL(file);
        processButton.disabled = false;
        downloadButton.style.display = 'none';
        status.textContent = 'Video loaded. Click Process to test basic functions.';
        progress.style.width = '0%';
        enhancedVideo.src = '';
    }

    // Event Listeners
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
    });

    // Process button click handler
    processButton.addEventListener('click', async () => {
        if (!currentFile) {
            status.textContent = 'Please select a video first';
            return;
        }
    
        try {
            // Disable UI during processing
            processButton.disabled = true;
            downloadButton.style.display = 'none';
            // Show progress container
            document.getElementById('progressContainer').style.display = 'block';
            progress.style.width = '0%';
            status.textContent = 'Processing video...';
    
            // Run basic functionality test with progress tracking
            const finalVideo = await processor.testBasicFunctionality(currentFile, (progressValue) => {
                // Update progress bar width with transition
                progress.style.width = `${progressValue}%`;
                
                // Update status text based on progress ranges
                if (progressValue <= 20) {
                    status.textContent = `Extracting frames (${Math.round(progressValue)}%)`;
                } else if (progressValue <= 30) {
                    status.textContent = `Extracting audio (${Math.round(progressValue)}%)`;
                } else if (progressValue <= 80) {
                    status.textContent = `Enhancing video (${Math.round(progressValue)}%)`;
                } else if (progressValue <= 90) {
                    status.textContent = `Merging frames (${Math.round(progressValue)}%)`;
                } else {
                    status.textContent = `Finalizing video (${Math.round(progressValue)}%)`;
                }
            });
    
            // Update UI with result
            enhancedVideo.src = URL.createObjectURL(finalVideo);
            status.textContent = 'Processing complete!';
            
            // Enable download
            downloadButton.style.display = 'inline-block';
            downloadButton.onclick = () => {
                const a = document.createElement('a');
                a.href = enhancedVideo.src;
                a.download = 'processed_' + currentFile.name;
                a.click();
            };
    
        } catch (error) {
            console.error('Test failed:', error);
            status.textContent = 'Error during processing: ' + error.message;
        } finally {
            processButton.disabled = false;
            // Hide progress container after completion
            setTimeout(() => {
                document.getElementById('progressContainer').style.display = 'none';
            }, 1000);
        }
    });

    // Video event listeners
    originalVideo.addEventListener('error', (e) => {
        console.error('Original video error:', e);
        status.textContent = 'Error loading original video';
    });

    enhancedVideo.addEventListener('error', (e) => {
        if (enhancedVideo.src && !enhancedVideo.src.endsWith('blob:null')) {
            console.error('Processed video error:', e);
            status.textContent = 'Error loading processed video';
        }
    });
});