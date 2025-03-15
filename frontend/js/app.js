document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const serverForm = document.getElementById('server-form');
    const modelSelect = document.getElementById('model-select');
    const maxModelLenSelect = document.getElementById('max-model-len');
    const speculativeDecodingCheckbox = document.getElementById('speculative-decoding');
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const restartBtn = document.getElementById('restart-btn');
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');
    const serverDetails = document.getElementById('server-details');
    const currentModelText = document.getElementById('current-model');
    const uptimeText = document.getElementById('uptime');
    const terminal = document.getElementById('terminal');
    const modelInfo = document.getElementById('model-info');
    const clearTerminalBtn = document.getElementById('clear-terminal');
    const copyTerminalBtn = document.getElementById('copy-terminal');
    const resetBtn = document.getElementById('reset-btn');
    const gpuMonitor = document.getElementById('gpu-monitor');
    const refreshGpuBtn = document.getElementById('refresh-gpu');
    const gpuTab = document.getElementById('gpu-tab');
    const liveGpuBtn = document.getElementById('live-gpu');
    
    // Toast elements
    const toast = new bootstrap.Toast(document.getElementById('toast'));
    const toastTitle = document.getElementById('toast-title');
    const toastMessage = document.getElementById('toast-message');
    
    // API Base URL - adjust if needed
    const API_BASE_URL = window.location.protocol + '//' + window.location.host;
    
    // WebSocket connection
    let socket = null;
    
    // Store model data
    let modelsData = {
        models: [],
        modelGroups: {}
    };
    
    // Server status data
    let serverStatus = {
        running: false,
        startTime: null,
        uptime: 0
    };
    
    // GPU monitoring variables
    let gpuMonitorInterval = null;
    const GPU_REFRESH_INTERVAL = 5000; // 5 seconds
    let isLiveMonitoring = false;
    
    // Add a variable to store server info
    let serverInfo = {
        hostname: window.location.hostname,
        username: 'user'
    };
    
    // Initialize the app
    init();
    
    function init() {
        // Initialize WebSocket connection
        connectWebSocket();
        
        // Load models from API
        fetchModels();
        
        // Check server status
        fetchServerStatus();
        
        // Get server hostname
        fetchServerHostname();
        
        // Get tmux session info
        fetchTmuxInfo();
        
        // Set up event listeners
        setupEventListeners();
        
        // Set up interval to update uptime
        setInterval(updateUptime, 1000);
        
        // Start GPU monitoring immediately
        fetchGpuStats();
        startGpuMonitoring();
        
        // Set up tab change event listeners
        setupTabListeners();
    }
    
    function setupEventListeners() {
        // Server form submission (start server)
        if (serverForm) {
            serverForm.addEventListener('submit', function(e) {
                e.preventDefault();
                if (serverForm.checkValidity()) {
                    startServer();
                }
            });
        }
        
        // Stop button
        if (stopBtn) {
            stopBtn.addEventListener('click', stopServer);
        }
        
        // Restart button
        if (restartBtn) {
            restartBtn.addEventListener('click', restartServer);
        }
        
        // Model select change
        if (modelSelect) {
            modelSelect.addEventListener('change', handleModelSelectChange);
        }
        
        // Clear terminal button
        if (clearTerminalBtn) {
            clearTerminalBtn.addEventListener('click', function() {
                if (terminal) {
                    terminal.innerHTML = '';
                }
            });
        }
        
        // Copy terminal button
        if (copyTerminalBtn) {
            copyTerminalBtn.addEventListener('click', function() {
                if (terminal) {
                    const text = terminal.innerText;
                    navigator.clipboard.writeText(text)
                        .then(() => showToast('Success', 'Terminal output copied to clipboard'))
                        .catch(err => showToast('Error', 'Failed to copy: ' + err));
                }
            });
        }
        
        // Reset button
        if (resetBtn) {
            resetBtn.addEventListener('click', resetUI);
        }
        
        // Refresh GPU button
        if (refreshGpuBtn) {
            refreshGpuBtn.addEventListener('click', fetchGpuStats);
        }
        
        // GPU tab click - start monitoring when tab is shown
        if (gpuTab) {
            gpuTab.addEventListener('shown.bs.tab', function() {
                fetchGpuStats();
                startGpuMonitoring();
            });
            
            // When switching away from GPU tab, stop the monitoring
            gpuTab.addEventListener('hidden.bs.tab', stopGpuMonitoring);
        }
        
        // Live GPU monitoring button
        if (liveGpuBtn) {
            liveGpuBtn.addEventListener('click', function() {
                isLiveMonitoring = !isLiveMonitoring;
                if (isLiveMonitoring) {
                    liveGpuBtn.innerHTML = '<i class="bi bi-pause-circle"></i> Pause Live Updates';
                    liveGpuBtn.classList.replace('btn-success', 'btn-warning');
                    // Start more frequent updates (every second)
                    stopGpuMonitoring();
                    gpuMonitorInterval = setInterval(fetchGpuStats, 1000);
                } else {
                    liveGpuBtn.innerHTML = '<i class="bi bi-play-circle"></i> Live Updates';
                    liveGpuBtn.classList.replace('btn-warning', 'btn-success');
                    // Return to normal update frequency
                    stopGpuMonitoring();
                    gpuMonitorInterval = setInterval(fetchGpuStats, GPU_REFRESH_INTERVAL);
                }
            });
        }
        
        // Add a button to check tmux sessions
        const checkTmuxBtn = document.createElement('button');
        checkTmuxBtn.className = 'btn btn-sm btn-info mt-2';
        checkTmuxBtn.innerHTML = '<i class="bi bi-terminal"></i> Check Tmux Sessions';
        checkTmuxBtn.onclick = checkTmuxSessions;
        
        // Add it after the server details
        if (serverDetails) {
            serverDetails.parentNode.appendChild(checkTmuxBtn);
            
            // Add WebSocket status indicator
            const wsStatusDiv = document.createElement('div');
            wsStatusDiv.className = 'd-flex justify-content-between mt-2';
            wsStatusDiv.innerHTML = `
                <span>WebSocket:</span>
                <span id="ws-status" class="badge bg-secondary">Connecting...</span>
            `;
            serverDetails.appendChild(wsStatusDiv);
        }
    }
    
    function setupTabListeners() {
        // Refresh tmux info when switching to tmux tab
        const tmuxTab = document.getElementById('tmux-tab');
        if (tmuxTab) {
            tmuxTab.addEventListener('shown.bs.tab', function() {
                fetchTmuxInfo();
            });
        }
    }
    
    function connectWebSocket() {
        // Close existing connection if any
        if (socket && socket.readyState !== WebSocket.CLOSED) {
            socket.close();
        }
        
        // Create a new WebSocket connection
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/api/ws`;
        
        console.log(`Connecting to WebSocket at: ${wsUrl}`);
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = function(event) {
            console.log('WebSocket connection established');
            appendToTerminal('WebSocket connection established', 'system');
            
            // Update connection status
            const statusIndicator = document.getElementById('ws-status');
            if (statusIndicator) {
                statusIndicator.className = 'badge bg-success';
                statusIndicator.textContent = 'Connected';
            }
            
            // Set up ping interval to keep connection alive
            if (window.pingInterval) {
                clearInterval(window.pingInterval);
            }
            window.pingInterval = setInterval(function() {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({type: 'ping'}));
                }
            }, 30000); // Send ping every 30 seconds
        };
        
        socket.onmessage = function(event) {
            try {
                // First try to parse as JSON
                let message;
                try {
                    message = JSON.parse(event.data);
                } catch (e) {
                    // If it's not JSON, treat it as plain text
                    appendToTerminal(event.data);
                    return;
                }
                
                // Skip pong messages completely
                if (message.type === "pong") {
                    return;
                }
                
                if (message.type === "output") {
                    // Extract just the data part
                    const outputText = message.data;
                    
                    // Handle special characters for progress bars
                    const formattedText = formatTerminalOutput(outputText);
                    
                    // Append to terminal
                    appendToTerminal(formattedText);
                    
                    // Check if this output indicates the server has started
                    if (formattedText.includes("Server is running") || 
                        formattedText.includes("Server started successfully")) {
                        // Update UI to show server is running
                        updateServerStatus({ running: true });
                        
                        // Explicitly enable the stop and restart buttons
                        stopBtn.disabled = false;
                        restartBtn.disabled = false;
                    }
                    
                    // Check if this output indicates the server has stopped
                    if (formattedText.includes("Server stopped") || 
                        formattedText.includes("Server is not running")) {
                        // Update UI to show server is not running
                        updateServerStatus({ running: false });
                        
                        // Explicitly disable the stop and restart buttons
                        stopBtn.disabled = true;
                        restartBtn.disabled = true;
                    }
                } else if (message.type === "status") {
                    // Handle status updates
                    updateUIForServerStatus(message.data);
                    
                    // Explicitly set button states based on running status
                    if (message.data.running) {
                        stopBtn.disabled = false;
                        restartBtn.disabled = false;
                        startBtn.disabled = true;
                    } else {
                        stopBtn.disabled = true;
                        restartBtn.disabled = true;
                        startBtn.disabled = false;
                    }
                } else if (message.type === "error") {
                    // Handle error messages
                    handleErrorMessage(message.data);
                }
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
                appendToTerminal(`Error processing message: ${error.message}`, "error");
            }
        };
        
        socket.onclose = function(event) {
            console.log('WebSocket connection closed');
            appendToTerminal('WebSocket connection closed. Reconnecting in 5 seconds...', 'system');
            
            // Update connection status
            const statusIndicator = document.getElementById('ws-status');
            if (statusIndicator) {
                statusIndicator.className = 'badge bg-danger';
                statusIndicator.textContent = 'Disconnected';
            }
            
            // Attempt to reconnect after a delay
            setTimeout(connectWebSocket, 5000);
        };
        
        socket.onerror = function(error) {
            console.error('WebSocket error:', error);
            appendToTerminal('WebSocket error occurred', 'error');
            
            // Update connection status
            const statusIndicator = document.getElementById('ws-status');
            if (statusIndicator) {
                statusIndicator.className = 'badge bg-warning';
                statusIndicator.textContent = 'Error';
            }
        };
    }
    
    function fetchModels() {
        // Make sure we're using the correct API endpoint
        fetch(`${API_BASE_URL}/api/models`)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                modelsData.models = data.models;
                
                // Group models by name
                modelsData.modelGroups = {};
                data.models.forEach(model => {
                    if (!modelsData.modelGroups[model.model_name]) {
                        modelsData.modelGroups[model.model_name] = [];
                    }
                    modelsData.modelGroups[model.model_name].push(model);
                });
                
                // Populate model select dropdown
                populateModelSelect();
            })
            .catch(error => {
                console.error('Error fetching models:', error);
                showToast('Error', 'Failed to load models. Please check the console for details.');
            });
    }
    
    function populateModelSelect() {
        // Clear existing options except the placeholder
        while (modelSelect.options.length > 1) {
            modelSelect.remove(1);
        }
        
        // Add model groups
        const modelGroups = modelsData.modelGroups || {};
        
        Object.keys(modelGroups).sort().forEach(modelName => {
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = modelName;
            modelSelect.appendChild(option);
        });
    }
    
    function handleModelSelectChange() {
        const selectedModel = modelSelect.value;
        
        // Clear max model length options
        maxModelLenSelect.innerHTML = '<option value="" selected>Default</option>';
        
        // Reset speculative decoding
        speculativeDecodingCheckbox.checked = false;
        
        if (!selectedModel) {
            modelInfo.innerHTML = '<p class="text-muted">Select a model to see details</p>';
            return;
        }
        
        // Get configurations for this model
        const modelConfigs = modelsData.modelGroups[selectedModel] || [];
        
        // Populate max model length options
        const uniqueLengths = [...new Set(modelConfigs.map(config => config.max_model_len))];
        uniqueLengths.sort((a, b) => a - b);
        
        uniqueLengths.forEach(length => {
            const option = document.createElement('option');
            option.value = length;
            option.textContent = length.toLocaleString();
            maxModelLenSelect.appendChild(option);
        });
        
        // Show model information
        updateModelInfo(selectedModel);
        
        // If there's only one max_model_len option, select it by default
        if (uniqueLengths.length === 1) {
            maxModelLenSelect.value = uniqueLengths[0];
        }
        
        // Check if any config has speculative decoding enabled
        const hasSpeculativeDecoding = modelConfigs.some(config => config.speculative_decoding);
        if (hasSpeculativeDecoding) {
            speculativeDecodingCheckbox.checked = true;
        }
    }
    
    function updateModelInfo(modelName) {
        const configs = modelsData.modelGroups[modelName] || [];
        
        if (configs.length === 0) {
            modelInfo.innerHTML = '<p class="text-muted">No information available</p>';
            return;
        }
        
        // Use the first config for basic info
        const baseConfig = configs[0];
        
        let html = `
            <h6>${modelName}</h6>
            <div class="mb-2">
                <strong>Path:</strong> ${baseConfig.model_path}
            </div>
            <div class="mb-2">
                <strong>Tensor Parallel Size:</strong> ${baseConfig.tensor_parallel_size}
            </div>
            <div class="mb-2">
                <strong>GPU Memory Utilization:</strong> ${baseConfig.gpu_memory_utilization || 'Default'}
            </div>
            <div class="mb-2">
                <strong>Enforce Eager:</strong> ${baseConfig.enforce_eager ? 'Yes' : 'No'}
            </div>
            <div class="mb-2">
                <strong>Chat Template:</strong> ${baseConfig.chat_template ? 'Yes' : 'No'}
            </div>
            <div class="mb-3">
                <strong>Usage Stats:</strong>
                <ul class="list-unstyled ms-3 mb-0">
                    <li>Generations: ${baseConfig.generations_count.toLocaleString()}</li>
                    <li>Tokens Generated: ${baseConfig.tokens_generated_count.toLocaleString()}</li>
                    <li>Avg. Tokens/sec: ${baseConfig.avg_tokens_per_second.toFixed(2)}</li>
                </ul>
            </div>
            
            <h6>Available Configurations:</h6>
        `;
        
        // Add each configuration
        configs.forEach(config => {
            html += `
                <div class="model-config-item">
                    <div><strong>Max Length:</strong> ${config.max_model_len.toLocaleString()}</div>
                    <div><strong>Speculative Decoding:</strong> ${config.speculative_decoding ? 'Yes' : 'No'}</div>
                    <div><strong>Max Batch Size:</strong> ${config.max_batch_size || 'Not set'}</div>
                </div>
            `;
        });
        
        modelInfo.innerHTML = html;
    }
    
    async function fetchServerStatus() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/status`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const status = await response.json();
            updateServerStatus(status);
        } catch (error) {
            console.error('Error fetching server status:', error);
        }
    }
    
    function updateServerStatus(status) {
        serverStatus = status;
        
        if (status.running) {
            statusIndicator.className = 'status-indicator status-running';
            statusText.textContent = 'Running';
            statusText.className = 'text-success';
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            restartBtn.disabled = false;
            
            // Show server details
            serverDetails.classList.remove('d-none');
            
            // Update current model
            if (status.model) {
                currentModelText.textContent = status.model;
            }
            
            // Set start time if provided
            if (status.startTime) {
                serverStatus.startTime = new Date(status.startTime);
                updateUptime();
            }
        } else {
            statusIndicator.className = 'status-indicator status-stopped';
            
            if (status.error) {
                statusText.textContent = 'Failed';
                statusText.className = 'text-danger';
            } else {
                statusText.textContent = 'Stopped';
                statusText.className = 'text-secondary';
            }
            
            startBtn.disabled = false;
            stopBtn.disabled = true;
            restartBtn.disabled = true;
            
            // Hide server details
            serverDetails.classList.add('d-none');
            
            // Reset uptime
            serverStatus.startTime = null;
            uptimeText.textContent = '00:00:00';
        }
    }
    
    function updateUptime() {
        if (!serverStatus.running) return;
        
        // Increment uptime
        serverStatus.uptime += 1;
        
        // Format uptime
        const seconds = Math.floor(serverStatus.uptime % 60);
        const minutes = Math.floor((serverStatus.uptime / 60) % 60);
        const hours = Math.floor(serverStatus.uptime / 3600);
        
        const formattedUptime = 
            (hours > 0 ? hours + 'h ' : '') + 
            (minutes > 0 ? minutes + 'm ' : '') + 
            seconds + 's';
        
        uptimeText.textContent = formattedUptime;
    }
    
    async function startServer() {
        const model = modelSelect.value;
        const maxModelLen = maxModelLenSelect.value ? parseInt(maxModelLenSelect.value) : null;
        const speculativeDecoding = speculativeDecodingCheckbox.checked;
        
        if (!model) {
            showToast('Error', 'Please select a model');
            return;
        }
        
        try {
            startBtn.disabled = true;
            startBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting...';
            
            const response = await fetch(`${API_BASE_URL}/api/server/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: model,
                    max_model_len: maxModelLen,
                    speculative_decoding: speculativeDecoding
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to start server');
            }
            
            const data = await response.json();
            appendToTerminal(`Starting server with model: ${model}`, 'system');
            
            if (maxModelLen) {
                appendToTerminal(`Max model length: ${maxModelLen}`, 'system');
            }
            
            if (speculativeDecoding) {
                appendToTerminal('Speculative decoding enabled', 'system');
            }
            
            showToast('Success', 'Server starting...');
            
            // Update UI to show server is starting
            statusIndicator.classList.remove('status-stopped');
            statusIndicator.classList.add('status-running');
            statusText.textContent = 'Starting...';
            
        } catch (error) {
            console.error('Error starting server:', error);
            showToast('Error', error.message);
            startBtn.disabled = false;
        } finally {
            startBtn.innerHTML = '<i class="bi bi-play-fill"></i> Start Server';
        }
    }
    
    async function stopServer() {
        try {
            stopBtn.disabled = true;
            stopBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Stopping...';
            
            // First, try to get the tmux session info to find the vLLM server session
            const tmuxResponse = await fetch(`${API_BASE_URL}/api/server/tmux-info`);
            if (tmuxResponse.ok) {
                const tmuxData = await tmuxResponse.json();
                if (tmuxData.vllm_server && tmuxData.vllm_server.exists && tmuxData.vllm_server.name) {
                    // Log that we're trying to stop the specific session
                    appendToTerminal(`Attempting to stop vLLM server in tmux session: ${tmuxData.vllm_server.name}`, 'system');
                    
                    // Try to send Ctrl+C to the tmux session first
                    try {
                        await fetch(`${API_BASE_URL}/api/server/kill-tmux-session`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ session: tmuxData.vllm_server.name })
                        });
                    } catch (e) {
                        console.error("Error killing tmux session:", e);
                    }
                }
            }
            
            // Now call the regular stop endpoint
            const response = await fetch(`${API_BASE_URL}/api/server/stop`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to stop server');
            }
            
            appendToTerminal('Stopping server...', 'system');
            showToast('Success', 'Server stopped');
            
            // Update UI immediately
            updateServerStatus({ running: false });
            
        } catch (error) {
            console.error('Error stopping server:', error);
            showToast('Error', error.message);
        } finally {
            stopBtn.innerHTML = '<i class="bi bi-stop-fill"></i> Stop Server';
            stopBtn.disabled = false;
        }
    }
    
    async function restartServer() {
        const model = modelSelect.value;
        const maxModelLen = maxModelLenSelect.value ? parseInt(maxModelLenSelect.value) : null;
        const speculativeDecoding = speculativeDecodingCheckbox.checked;
        
        if (!model) {
            showToast('Error', 'Please select a model');
            return;
        }
        
        try {
            restartBtn.disabled = true;
            restartBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Restarting...';
            
            const response = await fetch(`${API_BASE_URL}/api/server/restart`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model: model,
                    max_model_len: maxModelLen,
                    speculative_decoding: speculativeDecoding
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to restart server');
            }
            
            appendToTerminal(`Restarting server with model: ${model}`, 'system');
            showToast('Success', 'Server restarting...');
            
            // Update UI to show server is restarting
            statusIndicator.classList.remove('status-stopped');
            statusIndicator.classList.add('status-running');
            statusText.textContent = 'Restarting...';
            
        } catch (error) {
            console.error('Error restarting server:', error);
            showToast('Error', error.message);
        } finally {
            restartBtn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Restart Server';
            restartBtn.disabled = false;
        }
    }
    
    function appendToTerminal(text, type = "normal") {
        if (!terminal) return;
        
        // Check if the text contains progress bar indicators
        const isProgressBar = text.includes('|') && 
                             (text.includes('█') || 
                              text.includes('▏') || 
                              text.includes('▎') || 
                              text.includes('▍') || 
                              text.includes('▌') || 
                              text.includes('▋') || 
                              text.includes('▊') || 
                              text.includes('▉'));
        
        // If this is a progress bar update, replace the last line instead of adding a new one
        if (isProgressBar && terminal.lastChild && terminal.lastChild.classList.contains('progress-bar-line')) {
            terminal.lastChild.textContent = text;
            return;
        }
        
        const line = document.createElement('div');
        line.className = `terminal-line ${type}`;
        
        // Add progress-bar-line class if this is a progress bar
        if (isProgressBar) {
            line.classList.add('progress-bar-line');
        }
        
        line.textContent = text;
        terminal.appendChild(line);
        
        // Auto-scroll to bottom
        terminal.scrollTop = terminal.scrollHeight;
    }
    
    function showToast(title, message) {
        toastTitle.textContent = title;
        toastMessage.textContent = message;
        toast.show();
    }
    
    function resetUI() {
        // Close WebSocket connection and reconnect
        if (socket) {
            socket.close();
        }
        
        // Clear terminal
        terminal.innerHTML = '';
        
        // Reset UI state
        updateServerStatus({ running: false });
        
        // Reconnect WebSocket
        connectWebSocket();
        
        // Fetch server status again
        fetchServerStatus();
        
        showToast('Success', 'UI has been reset');
    }
    
    function startGpuMonitoring() {
        // Clear any existing interval
        stopGpuMonitoring();
        
        // Start new interval
        gpuMonitorInterval = setInterval(fetchGpuStats, GPU_REFRESH_INTERVAL);
    }
    
    function stopGpuMonitoring() {
        if (gpuMonitorInterval) {
            clearInterval(gpuMonitorInterval);
            gpuMonitorInterval = null;
        }
    }
    
    async function fetchGpuStats() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/gpu/stats`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            updateGpuMonitor(data);
        } catch (error) {
            console.error('Error fetching GPU stats:', error);
            gpuMonitor.innerHTML = `<div class="text-danger">Error fetching GPU stats: ${error.message}</div>`;
        }
    }
    
    function updateGpuMonitor(data) {
        // Clear the monitor
        gpuMonitor.innerHTML = '';
        
        if (!data.gpus || data.gpus.length === 0) {
            gpuMonitor.innerHTML = '<div class="text-warning">No GPU information available</div>';
            return;
        }
        
        // Add timestamp
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'gpu-timestamp';
        timestampDiv.innerHTML = `<span class="text-info">Last updated: ${new Date().toLocaleTimeString()}</span>`;
        gpuMonitor.appendChild(timestampDiv);
        
        // Create grid container for GPUs
        const gpuGrid = document.createElement('div');
        gpuGrid.className = 'gpu-grid';
        
        // Add each GPU
        data.gpus.forEach((gpu, index) => {
            const gpuDiv = document.createElement('div');
            gpuDiv.className = 'gpu-item';
            
            // Calculate memory usage percentage
            const memoryUsagePercent = (gpu.memory.used / gpu.memory.total) * 100;
            const utilizationColor = getUtilizationColor(gpu.utilization);
            const memoryColor = getUtilizationColor(memoryUsagePercent);
            
            // Format memory values with more precision
            const memUsedMB = gpu.memory.used.toFixed(0);
            const memTotalMB = gpu.memory.total.toFixed(0);
            const memUsedGB = (gpu.memory.used / 1024).toFixed(2);
            const memTotalGB = (gpu.memory.total / 1024).toFixed(2);
            
            gpuDiv.innerHTML = `
                <div class="gpu-header">
                    <strong class="text-light">GPU ${gpu.index}: ${gpu.name.split(' ').slice(-1)[0]}</strong>
                </div>
                <div class="d-flex justify-content-between">
                    <div class="gpu-stat">
                        <span>Temp:</span>
                        <span>${gpu.temperature}°C</span>
                    </div>
                    <div class="gpu-stat">
                        <span>Util:</span>
                        <span style="color: ${utilizationColor}">${gpu.utilization}%</span>
                    </div>
                </div>
                <div class="gpu-stat">
                    <span>Mem:</span>
                    <span style="color: ${memoryColor}">${memUsedMB}/${memTotalMB} MB (${memoryUsagePercent.toFixed(1)}%)</span>
                </div>
                <div class="gpu-stat">
                    <span>Mem (GB):</span>
                    <span style="color: ${memoryColor}">${memUsedGB}/${memTotalGB} GB</span>
                </div>
                <div class="gpu-progress">
                    <div class="progress" style="height: 4px;">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${memoryUsagePercent}%; background-color: ${memoryColor};" 
                             aria-valuenow="${memoryUsagePercent}" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                </div>
            `;
            
            gpuGrid.appendChild(gpuDiv);
        });
        
        gpuMonitor.appendChild(gpuGrid);
        
        // Add processes if available - in a more compact format
        if (data.processes && data.processes.length > 0) {
            const processesDiv = document.createElement('div');
            processesDiv.className = 'gpu-processes mt-2';
            
            let processesHtml = `
                <div class="small text-light mb-1">GPU Processes:</div>
                <div class="table-responsive">
                    <table class="table table-dark table-sm mb-0">
                        <thead>
                            <tr>
                                <th>GPU</th>
                                <th>PID</th>
                                <th>Process</th>
                                <th>Memory</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            data.processes.forEach(process => {
                // Truncate process name if too long
                const processName = process.name.length > 20 ? process.name.substring(0, 18) + '...' : process.name;
                const memoryMB = process.memory_usage.toFixed(0);
                const memoryGB = (process.memory_usage / 1024).toFixed(2);
                
                processesHtml += `
                    <tr>
                        <td>${process.gpu_id}</td>
                        <td>${process.pid}</td>
                        <td>${processName}</td>
                        <td>${memoryMB} MB (${memoryGB} GB)</td>
                    </tr>
                `;
            });
            
            processesHtml += `
                        </tbody>
                    </table>
                </div>
            `;
            
            processesDiv.innerHTML = processesHtml;
            gpuMonitor.appendChild(processesDiv);
        }
    }
    
    function getUtilizationColor(percentage) {
        if (percentage < 30) return '#28a745'; // Green
        if (percentage < 70) return '#ffc107'; // Yellow
        return '#dc3545'; // Red
    }
    
    function handleErrorMessage(errorData) {
        console.log("Handling error message:", errorData);
        
        // Create error alert with more compact styling
        const errorDiv = document.createElement('div');
        errorDiv.className = 'alert alert-danger mt-2'; // Reduced top margin
        
        if (errorData.known === true) {
            console.log("Displaying as known error");
            // Known error with more compact solution
            errorDiv.innerHTML = `
                <h5><i class="bi bi-exclamation-triangle-fill"></i> Known Error</h5>
                <p><strong>${errorData.message}</strong></p>
                <hr>
                <p><strong>Solution:</strong> ${errorData.solution}</p>
            `;
            
            // Show toast notification
            showToast('Known Error Detected', errorData.message);
        } else {
            console.log("Displaying as new error");
            // New error with more compact layout
            errorDiv.innerHTML = `
                <h5><i class="bi bi-exclamation-triangle-fill"></i> New Error</h5>
                <p><strong>${errorData.message}</strong></p>
                <hr>
                <p>This error has been added to the database. No solution is available yet.</p>
                <p>Error ID: ${errorData.error_id || 'Unknown'}</p>
            `;
            
            // Show toast notification
            showToast('New Error Detected', 'This error has been added to the database for tracking.');
        }
        
        // Add to terminal
        terminal.appendChild(errorDiv);
        terminal.scrollTop = terminal.scrollHeight;
        
        // Also scroll to make the error visible
        errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    
    // Add this function to fetch tmux session info
    async function fetchTmuxInfo() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/server/tmux-info`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            updateTmuxInfo(data);
            
            // If no vLLM server is running, show a message in the terminal
            if (!data.vllm_server.exists) {
                appendToTerminal("No vLLM server is currently running. Start a server to see its output here.", "system");
            }
        } catch (error) {
            console.error('Error fetching tmux info:', error);
            // Don't show error in terminal to avoid confusion
        }
    }
    
    // Update the updateTmuxInfo function to remove debug sections
    function updateTmuxInfo(data) {
        // Get the tmux info container
        let tmuxInfoContainer = document.getElementById('tmux-info-container');
        if (!tmuxInfoContainer) {
            console.error('Tmux info container not found');
            return;
        }
        
        // Build the HTML content
        let html = `
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h5 class="mb-0">Tmux Sessions</h5>
                <button class="btn btn-sm btn-outline-info" onclick="fetchTmuxInfo()">
                    <i class="bi bi-arrow-repeat"></i> Refresh
                </button>
            </div>
        `;
        
        // Add all sessions
        if (data.all_sessions && data.all_sessions.length > 0) {
            html += `<div class="list-group">`;
            
            data.all_sessions.forEach(session => {
                const isApiServer = session === data.llm_manager.name;
                const isVllmServer = session === data.vllm_server.name;
                const sessionType = isApiServer ? "API Server" : (isVllmServer ? "vLLM Server" : "Other");
                const badgeClass = isApiServer ? "bg-primary" : (isVllmServer ? "bg-success" : "bg-secondary");
                
                html += `
                    <div class="list-group-item bg-dark text-light d-flex justify-content-between align-items-center">
                        <div>
                            <span>${session}</span>
                            <span class="badge ${badgeClass} ms-2">${sessionType}</span>
                        </div>
                        <div class="d-flex">
                            <button class="btn btn-sm btn-outline-light" 
                                    onclick="window.copyTmuxCommand('tmux attach -t ${session}')">
                                <i class="bi bi-clipboard"></i> Copy Command
                            </button>
                            <button class="btn btn-sm btn-outline-info ms-1" 
                                    onclick="window.copyTmuxCommand('ssh ${serverInfo.username}@${serverInfo.hostname} -t tmux attach -t ${session}')">
                                <i class="bi bi-clipboard"></i> SSH
                            </button>
                            <button class="btn btn-sm btn-outline-danger ms-1" 
                                    onclick="window.killTmuxSession('${session}')">
                                <i class="bi bi-x-circle"></i> Kill
                            </button>
                        </div>
                    </div>
                `;
            });
            
            html += `</div>`;
        } else {
            html += '<div class="alert alert-info">No tmux sessions found</div>';
        }
        
        // Update the container
        tmuxInfoContainer.innerHTML = html;
    }
    
    // Make the copyTmuxCommand function globally accessible
    window.copyTmuxCommand = function(text) {
        console.log("Copying to clipboard:", text);
        
        // Check if the Clipboard API is available
        if (!navigator.clipboard) {
            console.warn("Clipboard API not available - using fallback method");
            
            // Fallback method using a temporary textarea element
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';  // Prevent scrolling to bottom
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();
            
            let successful = false;
            try {
                successful = document.execCommand('copy');
            } catch (err) {
                console.error('Fallback: Unable to copy', err);
            }
            
            document.body.removeChild(textarea);
            
            if (successful) {
                console.log("Copy successful (fallback method)");
                showToast('Copied', 'Command copied to clipboard');
            } else {
                console.error("Copy failed (fallback method)");
                showToast('Error', 'Failed to copy to clipboard - please copy manually');
                // Show the text in a modal or alert for manual copying
                alert('Please copy this command manually: ' + text);
            }
            return;
        }
        
        // Use the Clipboard API if available
        navigator.clipboard.writeText(text)
            .then(() => {
                console.log("Copy successful");
                showToast('Copied', 'Command copied to clipboard');
            })
            .catch(err => {
                console.error('Failed to copy:', err);
                showToast('Error', 'Failed to copy to clipboard: ' + err);
                
                // Show the text in a modal or alert for manual copying
                alert('Please copy this command manually: ' + text);
            });
    };
    
    // Add this function to fetch server hostname
    async function fetchServerHostname() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/server/hostname`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            serverInfo = data;
        } catch (error) {
            console.error('Error fetching server hostname:', error);
        }
    }
    
    // Update the checkTmuxSessions function to switch to the terminal tab
    async function checkTmuxSessions() {
        try {
            // Switch to terminal tab to show output
            const terminalTab = document.getElementById('terminal-tab');
            if (terminalTab) {
                const tabInstance = new bootstrap.Tab(terminalTab);
                tabInstance.show();
            }
            
            const response = await fetch(`${API_BASE_URL}/api/server/check-tmux`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // The output will be sent to the terminal via WebSocket
            appendToTerminal("Checking tmux sessions...", "system");
        } catch (error) {
            console.error('Error checking tmux sessions:', error);
            appendToTerminal(`Error checking tmux sessions: ${error.message}`, "error");
        }
    }
    
    // Update the reconnectTmuxSession function to switch to the terminal tab
    async function reconnectTmuxSession() {
        try {
            // Switch to terminal tab to show output
            const terminalTab = document.getElementById('terminal-tab');
            if (terminalTab) {
                const tabInstance = new bootstrap.Tab(terminalTab);
                tabInstance.show();
            }
            
            appendToTerminal("Attempting to reconnect to tmux session...", "system");
            
            const response = await fetch(`${API_BASE_URL}/api/server/reconnect-tmux`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === "ok") {
                appendToTerminal(`Successfully reconnected to session: ${data.session}`, "success");
            } else {
                appendToTerminal(`Failed to reconnect: ${data.message}`, "error");
            }
        } catch (error) {
            console.error('Error reconnecting to tmux session:', error);
            appendToTerminal(`Error reconnecting to tmux session: ${error.message}`, "error");
        }
    }
    
    // Update the testTmuxOutput function to switch to the terminal tab
    async function testTmuxOutput() {
        try {
            // Switch to terminal tab to show output
            const terminalTab = document.getElementById('terminal-tab');
            if (terminalTab) {
                const tabInstance = new bootstrap.Tab(terminalTab);
                tabInstance.show();
            }
            
            appendToTerminal("Testing tmux output streaming...", "system");
            
            const response = await fetch(`${API_BASE_URL}/api/server/test-tmux-output`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === "ok") {
                appendToTerminal(`Test initiated for session: ${data.session}`, "system");
            } else {
                appendToTerminal(`Failed to test: ${data.message}`, "error");
            }
        } catch (error) {
            console.error('Error testing tmux output:', error);
            appendToTerminal(`Error testing tmux output: ${error.message}`, "error");
        }
    }
    
    // Update the debugTmux function to switch to the terminal tab
    async function debugTmux() {
        try {
            // Switch to terminal tab to show output
            const terminalTab = document.getElementById('terminal-tab');
            if (terminalTab) {
                const tabInstance = new bootstrap.Tab(terminalTab);
                tabInstance.show();
            }
            
            appendToTerminal("Debugging tmux sessions...", "system");
            
            const response = await fetch(`${API_BASE_URL}/api/server/debug-tmux`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            appendToTerminal("Tmux Debug Results:", "system");
            appendToTerminal(`tmux ls return code: ${data.tmux_ls_returncode}`, "system");
            appendToTerminal(`tmux ls output: ${data.tmux_ls_output || 'No output'}`, "system");
            appendToTerminal(`Session file exists: ${data.session_file_exists}`, "system");
            appendToTerminal(`Session name: ${data.session_name || 'None'}`, "system");
        } catch (error) {
            console.error('Error debugging tmux:', error);
            appendToTerminal(`Error debugging tmux: ${error.message}`, "error");
        }
    }
    
    function updateUIForServerStatus(status) {
        if (status.running) {
            // Server is running
            statusIndicator.className = 'status-indicator status-running';
            statusText.textContent = 'Running';
            startBtn.disabled = true;
            stopBtn.disabled = false;
            restartBtn.disabled = false;
            
            // Update server details
            serverDetails.style.display = 'block';
            currentModelText.textContent = status.model || 'Unknown';
            
            // Update uptime
            serverStatus.running = true;
            serverStatus.startTime = new Date(status.start_time);
            updateUptime();
            
            appendToTerminal(`Server is running with model: ${status.model}`, 'system');
        } else {
            // Server is not running
            statusIndicator.className = 'status-indicator status-stopped';
            statusText.textContent = 'Stopped';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            restartBtn.disabled = true;
            
            // Hide server details
            serverDetails.style.display = 'none';
            
            // Reset uptime
            serverStatus.running = false;
            serverStatus.startTime = null;
            uptimeText.textContent = '00:00:00';
            
            appendToTerminal('Server is not running. Use the form on the left to start one.', 'system');
        }
    }

    function updateTerminalForNoServer() {
        if (!serverStatus.running && terminal) {
            // Add a message and button to the terminal
            const noServerDiv = document.createElement('div');
            noServerDiv.className = 'text-center mt-4';
            noServerDiv.innerHTML = `
                <p class="text-light mb-3">No LLM server is currently running.</p>
                <button class="btn btn-primary" onclick="scrollToServerForm()">
                    <i class="bi bi-play-fill"></i> Start a Server
                </button>
            `;
            terminal.appendChild(noServerDiv);
        }
    }

    function scrollToServerForm() {
        // Scroll to the server form
        serverForm.scrollIntoView({ behavior: 'smooth' });
        // Highlight the form briefly
        serverForm.classList.add('highlight-form');
        setTimeout(() => {
            serverForm.classList.remove('highlight-form');
        }, 2000);
    }

    // Add this to your CSS
    const style = document.createElement('style');
    style.textContent = `
        .highlight-form {
            box-shadow: 0 0 15px rgba(0, 123, 255, 0.7);
            transition: box-shadow 0.5s ease-in-out;
        }
    `;
    document.head.appendChild(style);

    // Function to format terminal output with proper Unicode handling
    function formatTerminalOutput(text) {
        // Replace Unicode escape sequences with actual characters
        return text.replace(/\\u([0-9a-fA-F]{4})/g, (match, hex) => {
            return String.fromCharCode(parseInt(hex, 16));
        });
    }

    // Make the killTmuxSession function globally accessible
    window.killTmuxSession = async function(sessionName) {
        // Show confirmation dialog
        if (!confirm(`Are you sure you want to kill the tmux session "${sessionName}"?`)) {
            return;
        }
        
        try {
            // Switch to terminal tab to show output
            const terminalTab = document.getElementById('terminal-tab');
            if (terminalTab) {
                const tabInstance = new bootstrap.Tab(terminalTab);
                tabInstance.show();
            }
            
            appendToTerminal(`Attempting to kill tmux session: ${sessionName}...`, "system");
            
            // Create a new endpoint in the API to kill a tmux session
            const response = await fetch(`${API_BASE_URL}/api/server/kill-tmux-session`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ session: sessionName })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.status === "ok") {
                appendToTerminal(`Successfully killed session: ${sessionName}`, "success");
                // Refresh the tmux info
                fetchTmuxInfo();
            } else {
                appendToTerminal(`Failed to kill session: ${data.message}`, "error");
            }
        } catch (error) {
            console.error('Error killing tmux session:', error);
            appendToTerminal(`Error killing tmux session: ${error.message}`, "error");
        }
    };
}); 