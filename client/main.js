async function sendRequest() {
    const url = document.getElementById('url').value;
    const port = document.getElementById('port').value;
    const model = document.getElementById('model').value;
    const task = document.getElementById('task').value;
    const apiKey = document.getElementById('apiKey').value;
    const work_type = document.getElementById('work_type').value;
    const procedure = document.getElementById('procedure').value;
    const requestType = document.getElementById('requestType').value;
    const responseDiv = document.getElementById('response');
    const spinner = document.getElementById('spinner');
    const button = document.getElementById('submitBtn');
    
    try {
        // 로딩 시작
        spinner.style.display = 'inline-block';
        button.disabled = true;
        responseDiv.textContent = '분석 중입니다...';
        
        const endpoint = `${url}:${port}/v1/${model}/${task}/${requestType}`;
        
        if (requestType === 'stream') {
            // SSE 연결 설정
            const eventSource = new EventSource(endpoint + '?' + new URLSearchParams({
                work_type,
                procedure
            }));
            
            eventSource.onmessage = (event) => {
                responseDiv.textContent += event.data + '\n';
            };
            
            eventSource.onerror = () => {
                eventSource.close();
                spinner.style.display = 'none';
                button.disabled = false;
            };
            
            return;
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'x-api-key': apiKey,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "input": {
                    "work_type": work_type,
                    "procedure": procedure
                }
            })
        });
        
        const data = await response.json();
        responseDiv.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        responseDiv.textContent = `Error: ${error.message}`;
    } finally {
        if (requestType !== 'stream') {
            spinner.style.display = 'none';
            button.disabled = false;
        }
    }
}