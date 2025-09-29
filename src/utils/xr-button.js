// minimal XRButton helper (adapted for our simple demo)
import * as THREE from "three";

export const ARButton = {
  createButton: function(renderer) {
    const button = document.createElement('button');
    button.style.position = 'absolute';
    button.style.right = '20px';
    button.style.top = '20px';
    button.style.padding = '8px 12px';
    button.style.background = '#0b84ff';
    button.style.color = '#fff';
    button.style.border = 'none';
    button.style.borderRadius = '6px';
    button.style.fontSize = '14px';
    button.style.fontWeight = '600';
    button.style.cursor = 'pointer';
    button.style.zIndex = '1000';
    button.setAttribute('data-testid', 'enter-ar');
    button.setAttribute('aria-label', 'Enter Augmented Reality mode');
    button.textContent = '🥽 Enter AR';
    
    button.onclick = async () => {
      try {
        if (!navigator.xr) {
          throw new Error('WebXR not supported on this device');
        }
        
        // Check if AR is supported
        const isSupported = await navigator.xr.isSessionSupported('immersive-ar');
        if (!isSupported) {
          throw new Error('AR not supported on this device');
        }
        
        button.textContent = 'Starting AR...';
        button.disabled = true;
        
        const session = await navigator.xr.requestSession('immersive-ar', { 
          requiredFeatures: ['hit-test'], 
          optionalFeatures: ['dom-overlay'],
          domOverlay: { root: document.body } 
        });
        
        renderer.xr.setSession(session);
        button.textContent = '🚪 Exit AR';
        button.disabled = false;
        button.style.background = '#dc2626';  // Red for exit
        
        session.addEventListener('end', () => {
          button.textContent = '🥽 Enter AR';
          button.style.background = '#0b84ff';  // Blue for enter
          button.disabled = false;
        });
        
      } catch (err) {
        console.error('AR Error:', err);
        button.textContent = '🥽 Enter AR';
        button.disabled = false;
        
        // More user-friendly error messages
        let message = 'AR not available';
        if (err.message.includes('not supported')) {
          message = 'AR requires a mobile device with ARCore (Android) or ARKit (iOS) support';
        } else if (err.message.includes('permissions')) {
          message = 'Please allow camera permissions for AR';
        } else if (err.message.includes('session')) {
          message = 'Unable to start AR session. Try refreshing the page.';
        }
        
        // Show a more elegant error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.innerHTML = `
          <div style="
            position: fixed; 
            top: 20px; 
            left: 50%; 
            transform: translateX(-50%); 
            background: #fee2e2; 
            color: #dc2626; 
            padding: 12px 20px; 
            border-radius: 8px; 
            border: 1px solid #fca5a5;
            z-index: 10000;
            font-size: 14px;
            font-weight: 500;
            max-width: 400px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
          ">
            ⚠️ ${message}
            <div style="font-size: 12px; margin-top: 4px; opacity: 0.8;">
              Try using Chrome or Safari on a mobile device
            </div>
          </div>
        `;
        document.body.appendChild(errorDiv);
        
        // Auto remove error after 5 seconds
        setTimeout(() => {
          if (document.body.contains(errorDiv)) {
            document.body.removeChild(errorDiv);
          }
        }, 5000);
      }
    };
    
    return button;
  }
}