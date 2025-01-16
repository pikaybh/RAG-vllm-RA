document.getElementById('postForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const resource = document.getElementById('resource').value;
    const data = document.getElementById('data').value;

    const responseDiv = document.getElementById('response');
    responseDiv.textContent = 'Sending request...';

    try {
        const response = await fetch(`/${resource}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: data,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const responseData = await response.json();
        responseDiv.textContent = JSON.stringify(responseData, null, 2);
    } catch (error) {
        responseDiv.textContent = `Error: ${error.message}`;
    }
});
