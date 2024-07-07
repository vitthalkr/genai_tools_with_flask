document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');
    const uploadForm = document.getElementById('uploadForm');
    const downloadSection = document.getElementById('downloadSection');
    const progressContainer = document.getElementById('progress-container');
    const progressCircle = document.getElementById('progress-circle');
    const progressText = document.getElementById('progress-text');
    let files = [];

    fileInput.addEventListener('change', (event) => {
        for (let file of event.target.files) {
            files.push(file);
            displayFile(file);
        }
        fileInput.value = ''; // Clear the input for new file
    });

    function displayFile(file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerText = file.name;

        const removeButton = document.createElement('button');
        removeButton.className = 'remove-button';
        removeButton.innerText = 'Remove';
        removeButton.addEventListener('click', () => {
            files = files.filter(f => f !== file);
            fileList.removeChild(fileItem);
        });

        fileItem.appendChild(removeButton);
        fileList.appendChild(fileItem);
    }

    uploadForm.addEventListener('submit', (event) => {
        event.preventDefault();
        if (files.length === 0) {
            alert('Please add at least one file.');
            return;
        }

        const formData = new FormData();
        files.forEach(file => formData.append('files', file));

        progressContainer.style.display = 'block';
        let percentCompleted = 0;

        const interval = setInterval(() => {
            percentCompleted += 1;
            progressText.innerText = `${percentCompleted}%`;
            progressCircle.style.background = `conic-gradient(#4d5bf9 ${percentCompleted * 3.6}deg, #cadcff ${percentCompleted * 3.6}deg 360deg)`;

            if (percentCompleted >= 100) {
                clearInterval(interval);
            }
        }, 100);

        fetch('/summarize_emails', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            clearInterval(interval);
            progressText.innerText = '100%';
            progressCircle.style.background = `conic-gradient(#4d5bf9 360deg, #cadcff 0deg 360deg)`;

            downloadSection.innerHTML = ''; // Clear previous links
            console.log('Received data:', data); // Debugging log

            if (data.filenames && data.filenames.length > 0) {
                console.log(`Filenames received: ${data.filenames}`);
                data.filenames.forEach(filename => {
                    const downloadLink = document.createElement('a');
                    downloadLink.href = `/download?filename=${encodeURIComponent(filename)}`;
                    downloadLink.innerText = `Download ${filename}`;
                    downloadLink.className = 'download-button';
                    downloadLink.target = '_blank';
                    downloadLink.addEventListener('click', () => {
                        fetch(`/delete?filename=${encodeURIComponent(filename)}`)
                            .then(response => {
                                if (!response.ok) {
                                    throw new Error(`Failed to schedule deletion for file ${filename}`);
                                }
                                console.log(`Scheduled deletion for file ${filename}`);
                                // Optionally remove the download link from UI
                                downloadSection.removeChild(downloadLink.parentNode); // Remove the parent node (br included)
                            })
                            .catch(error => {
                                console.error('Error scheduling deletion:', error);
                                // Handle error
                            });
                    });
                    downloadSection.appendChild(downloadLink);
                    downloadSection.appendChild(document.createElement('br'));
                });
                downloadSection.style.display = 'block';
            } else {
                console.error('Error: Filenames are undefined or empty');
                alert('An error occurred while processing the files.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing the files.');
        });
    });
});
