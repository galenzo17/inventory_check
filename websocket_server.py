#!/usr/bin/env python3
"""
WebSocket Server for Real-time Medical Inventory Detection
Supports real-time video streaming and live detection updates
"""

import asyncio
import json
import time
import uuid
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
import logging

import torch
from PIL import Image
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, WebSocketException

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}
        self.processing_sessions: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, connection_info: Dict):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_metadata[client_id] = {
            **connection_info,
            'connected_at': datetime.utcnow().isoformat(),
            'last_heartbeat': time.time(),
            'message_count': 0
        }
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.connection_metadata:
            del self.connection_metadata[client_id]
        if client_id in self.processing_sessions:
            del self.processing_sessions[client_id]
        logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, message: Dict, client_id: str):
        """Send message to specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(json.dumps(message))
                self.connection_metadata[client_id]['message_count'] += 1
                return True
            except (ConnectionClosed, WebSocketException) as e:
                logger.warning(f"Failed to send to {client_id}: {e}")
                self.disconnect(client_id)
                return False
        return False
    
    async def broadcast(self, message: Dict, exclude_clients: Set[str] = None):
        """Broadcast message to all clients"""
        exclude_clients = exclude_clients or set()
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            if client_id not in exclude_clients:
                try:
                    await websocket.send_text(json.dumps(message))
                    self.connection_metadata[client_id]['message_count'] += 1
                except (ConnectionClosed, WebSocketException):
                    disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            'total_connections': len(self.active_connections),
            'active_sessions': len(self.processing_sessions),
            'connections': [
                {
                    'client_id': client_id,
                    'connected_at': metadata['connected_at'],
                    'message_count': metadata['message_count'],
                    'last_heartbeat': datetime.fromtimestamp(metadata['last_heartbeat']).isoformat()
                }
                for client_id, metadata in self.connection_metadata.items()
            ]
        }

class RealTimeDetector:
    """Real-time object detection processor"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.load_models()
        
        # Processing configurations
        self.config_presets = {
            'low_latency': {'max_fps': 60, 'input_size': 320, 'batch_size': 1},
            'balanced': {'max_fps': 30, 'input_size': 640, 'batch_size': 2},
            'high_quality': {'max_fps': 15, 'input_size': 1280, 'batch_size': 1}
        }
    
    def load_models(self):
        """Load detection models"""
        # Placeholder - would load actual models
        logger.info("Loading real-time detection models...")
        self.models['medical_fast'] = None  # Fast model for real-time
        self.models['medical_accurate'] = None  # Accurate model
    
    async def process_frame(self, frame_data: bytes, config: Dict) -> Dict:
        """Process single frame for detection"""
        try:
            start_time = time.time()
            
            # Decode frame
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {'error': 'Invalid frame data'}
            
            # Resize frame based on config
            input_size = config.get('input_size', 640)
            frame_resized = cv2.resize(frame, (input_size, input_size))
            
            # Run detection (placeholder)
            detections = await self.run_detection(frame_resized, config)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'processing_time': processing_time,
                'frame_size': frame.shape[:2],
                'detections': detections,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return {'error': str(e)}
    
    async def run_detection(self, frame: np.ndarray, config: Dict) -> List[Dict]:
        """Run object detection on frame"""
        # Placeholder detection logic
        # In real implementation, this would use the loaded models
        
        # Simulate some detections
        detections = [
            {
                'class_id': 0,
                'class_name': 'syringe',
                'confidence': 0.85,
                'bbox': [100, 100, 200, 200],
                'center': [150, 150]
            },
            {
                'class_id': 1,
                'class_name': 'bandage',
                'confidence': 0.92,
                'bbox': [300, 150, 400, 250],
                'center': [350, 200]
            }
        ]
        
        # Add some randomness to simulate real detections
        import random
        if random.random() > 0.7:  # 30% chance of detection
            return detections
        return []

class StreamProcessor:
    """Handles video stream processing"""
    
    def __init__(self, detector: RealTimeDetector, connection_manager: ConnectionManager):
        self.detector = detector
        self.connection_manager = connection_manager
        self.active_streams: Dict[str, asyncio.Task] = {}
    
    async def start_stream_processing(self, client_id: str, stream_config: Dict):
        """Start processing stream for client"""
        if client_id in self.active_streams:
            await self.stop_stream_processing(client_id)
        
        # Create processing task
        task = asyncio.create_task(
            self.process_stream(client_id, stream_config)
        )
        self.active_streams[client_id] = task
        
        # Store session info
        self.connection_manager.processing_sessions[client_id] = {
            'config': stream_config,
            'started_at': datetime.utcnow().isoformat(),
            'frame_count': 0,
            'detection_count': 0
        }
        
        await self.connection_manager.send_personal_message(
            {
                'type': 'stream_started',
                'config': stream_config,
                'timestamp': datetime.utcnow().isoformat()
            },
            client_id
        )
    
    async def stop_stream_processing(self, client_id: str):
        """Stop processing stream for client"""
        if client_id in self.active_streams:
            task = self.active_streams[client_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self.active_streams[client_id]
        
        if client_id in self.connection_manager.processing_sessions:
            session = self.connection_manager.processing_sessions[client_id]
            del self.connection_manager.processing_sessions[client_id]
            
            await self.connection_manager.send_personal_message(
                {
                    'type': 'stream_stopped',
                    'session_stats': session,
                    'timestamp': datetime.utcnow().isoformat()
                },
                client_id
            )
    
    async def process_stream(self, client_id: str, config: Dict):
        """Process continuous stream"""
        try:
            session = self.connection_manager.processing_sessions[client_id]
            max_fps = config.get('max_fps', 30)
            frame_interval = 1.0 / max_fps
            
            while client_id in self.connection_manager.active_connections:
                start_time = time.time()
                
                # In real implementation, this would process actual video frames
                # For now, we'll simulate periodic detection updates
                
                # Simulate frame processing
                if session['frame_count'] % 10 == 0:  # Send updates every 10 frames
                    detections = await self.simulate_detection_update()
                    
                    if detections:
                        session['detection_count'] += len(detections)
                        
                        await self.connection_manager.send_personal_message(
                            {
                                'type': 'detection_update',
                                'frame_number': session['frame_count'],
                                'detections': detections,
                                'counts': self.count_objects(detections),
                                'timestamp': datetime.utcnow().isoformat()
                            },
                            client_id
                        )
                
                session['frame_count'] += 1
                
                # Frame rate control
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.info(f"Stream processing cancelled for {client_id}")
        except Exception as e:
            logger.error(f"Stream processing error for {client_id}: {e}")
            await self.connection_manager.send_personal_message(
                {
                    'type': 'stream_error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                },
                client_id
            )
    
    async def simulate_detection_update(self) -> List[Dict]:
        """Simulate detection updates"""
        import random
        
        if random.random() > 0.6:  # 40% chance of detection
            return [
                {
                    'class_name': random.choice(['syringe', 'bandage', 'medicine_bottle']),
                    'confidence': round(random.uniform(0.7, 0.95), 2),
                    'bbox': [
                        random.randint(50, 200),
                        random.randint(50, 200),
                        random.randint(250, 400),
                        random.randint(250, 400)
                    ],
                    'id': str(uuid.uuid4())[:8]
                }
                for _ in range(random.randint(1, 3))
            ]
        return []
    
    def count_objects(self, detections: List[Dict]) -> Dict[str, int]:
        """Count objects by class"""
        counts = {}
        for detection in detections:
            class_name = detection['class_name']
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts

# FastAPI WebSocket Application
app = FastAPI(title="Medical Inventory WebSocket API")

# Initialize components
connection_manager = ConnectionManager()
detector = RealTimeDetector()
stream_processor = StreamProcessor(detector, connection_manager)

@app.get("/")
async def get_websocket_page():
    """Serve WebSocket test page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Inventory Detection WebSocket</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .connected { background-color: #d4edda; color: #155724; }
            .disconnected { background-color: #f8d7da; color: #721c24; }
            .messages { height: 400px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Medical Inventory Detection WebSocket Test</h1>
            <div id="status" class="status disconnected">Disconnected</div>
            
            <div class="controls">
                <button onclick="connect()">Connect</button>
                <button onclick="disconnect()">Disconnect</button>
                <button onclick="startStream()">Start Stream</button>
                <button onclick="stopStream()">Stop Stream</button>
                <button onclick="getStats()">Get Stats</button>
            </div>
            
            <div id="messages" class="messages"></div>
        </div>

        <script>
            let websocket = null;
            const messages = document.getElementById('messages');
            const status = document.getElementById('status');

            function connect() {
                websocket = new WebSocket('ws://localhost:8000/ws');
                
                websocket.onopen = function(event) {
                    status.textContent = 'Connected';
                    status.className = 'status connected';
                    addMessage('Connected to WebSocket');
                    
                    // Send connection info
                    websocket.send(JSON.stringify({
                        action: 'connect',
                        auth_token: 'demo_token',
                        stream_config: {
                            model: 'medical_fast',
                            confidence: 0.7,
                            max_fps: 30,
                            tracking: true
                        }
                    }));
                };

                websocket.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    addMessage(`Received: ${JSON.stringify(data, null, 2)}`);
                };

                websocket.onclose = function(event) {
                    status.textContent = 'Disconnected';
                    status.className = 'status disconnected';
                    addMessage('Disconnected from WebSocket');
                };

                websocket.onerror = function(error) {
                    addMessage(`Error: ${error}`);
                };
            }

            function disconnect() {
                if (websocket) {
                    websocket.close();
                }
            }

            function startStream() {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        action: 'start_stream',
                        config: {
                            mode: 'balanced',
                            max_fps: 30,
                            tracking: true
                        }
                    }));
                }
            }

            function stopStream() {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        action: 'stop_stream'
                    }));
                }
            }

            function getStats() {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        action: 'get_stats'
                    }));
                }
            }

            function addMessage(message) {
                const messageDiv = document.createElement('div');
                messageDiv.innerHTML = `<strong>${new Date().toLocaleTimeString()}</strong>: ${message}`;
                messages.appendChild(messageDiv);
                messages.scrollTop = messages.scrollHeight;
            }

            // Auto-connect on page load
            // connect();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint"""
    client_id = str(uuid.uuid4())
    
    try:
        await connection_manager.connect(
            websocket, 
            client_id, 
            {'user_agent': websocket.headers.get('user-agent', 'Unknown')}
        )
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(heartbeat_handler(client_id))
        
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                await handle_websocket_message(client_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError as e:
                await connection_manager.send_personal_message(
                    {
                        'type': 'error',
                        'message': 'Invalid JSON format',
                        'error': str(e)
                    },
                    client_id
                )
            except Exception as e:
                logger.error(f"WebSocket error for {client_id}: {e}")
                await connection_manager.send_personal_message(
                    {
                        'type': 'error',
                        'message': 'Internal server error',
                        'error': str(e)
                    },
                    client_id
                )
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Cleanup
        heartbeat_task.cancel()
        await stream_processor.stop_stream_processing(client_id)
        connection_manager.disconnect(client_id)

async def handle_websocket_message(client_id: str, message: Dict):
    """Handle incoming WebSocket messages"""
    action = message.get('action')
    
    if action == 'connect':
        await connection_manager.send_personal_message(
            {
                'type': 'connection_ack',
                'client_id': client_id,
                'timestamp': datetime.utcnow().isoformat()
            },
            client_id
        )
    
    elif action == 'start_stream':
        config = message.get('config', {})
        await stream_processor.start_stream_processing(client_id, config)
    
    elif action == 'stop_stream':
        await stream_processor.stop_stream_processing(client_id)
    
    elif action == 'get_stats':
        stats = connection_manager.get_connection_stats()
        await connection_manager.send_personal_message(
            {
                'type': 'stats',
                'data': stats,
                'timestamp': datetime.utcnow().isoformat()
            },
            client_id
        )
    
    elif action == 'ping':
        await connection_manager.send_personal_message(
            {
                'type': 'pong',
                'timestamp': datetime.utcnow().isoformat()
            },
            client_id
        )
    
    elif action == 'process_frame':
        # Handle single frame processing
        frame_data = message.get('frame_data')
        config = message.get('config', {})
        
        if frame_data:
            # Process frame (would need to decode base64 frame data)
            result = await detector.process_frame(frame_data, config)
            await connection_manager.send_personal_message(
                {
                    'type': 'frame_result',
                    'data': result,
                    'timestamp': datetime.utcnow().isoformat()
                },
                client_id
            )
    
    else:
        await connection_manager.send_personal_message(
            {
                'type': 'error',
                'message': f'Unknown action: {action}'
            },
            client_id
        )

async def heartbeat_handler(client_id: str):
    """Handle heartbeat for client"""
    try:
        while client_id in connection_manager.active_connections:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            
            success = await connection_manager.send_personal_message(
                {
                    'type': 'heartbeat',
                    'timestamp': datetime.utcnow().isoformat()
                },
                client_id
            )
            
            if not success:
                break
                
    except asyncio.CancelledError:
        pass

@app.get("/api/v1/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return connection_manager.get_connection_stats()

@app.get("/api/v1/ws/connections/{client_id}")
async def get_connection_info(client_id: str):
    """Get specific connection information"""
    if client_id not in connection_manager.connection_metadata:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    return connection_manager.connection_metadata[client_id]

if __name__ == "__main__":
    uvicorn.run(
        "websocket_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )