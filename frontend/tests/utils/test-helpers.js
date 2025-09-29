/**
 * Test utilities for AR functionality testing
 */

export class ARTestHelpers {
  constructor(page) {
    this.page = page;
  }

  /**
   * Mock WebXR support
   */
  async mockWebXRSupport(supported = true) {
    await this.page.addInitScript((isSupported) => {
      if (isSupported) {
        window.navigator.xr = {
          isSessionSupported: () => Promise.resolve(true),
          requestSession: () => Promise.resolve({
            addEventListener: () => {},
            removeEventListener: () => {},
            requestReferenceSpace: () => Promise.resolve({
              getOffsetReferenceSpace: () => ({})
            }),
            requestAnimationFrame: () => {},
            end: () => Promise.resolve(),
            renderState: {},
            inputSources: []
          })
        };
      } else {
        delete window.navigator.xr;
      }
    }, supported);
  }

  /**
   * Mock a generated Kolam pattern
   */
  async mockPatternGeneration(patternData = null) {
    const defaultPattern = {
      svgContent: `
        <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
          <circle cx="100" cy="100" r="50" fill="none" stroke="#8B5CF6" stroke-width="2"/>
          <path d="M60,100 Q100,60 140,100 Q100,140 60,100" fill="none" stroke="#8B5CF6" stroke-width="2"/>
        </svg>
      `,
      patternId: 'test-pattern-' + Date.now(),
      name: 'Test Kolam Pattern',
      timestamp: new Date().toISOString()
    };

    const pattern = patternData || defaultPattern;

    await this.page.evaluate((patternData) => {
      window.testPattern = patternData;
      window.dispatchEvent(new CustomEvent('patternGenerated', {
        detail: patternData
      }));
    }, pattern);

    return pattern;
  }

  /**
   * Wait for Three.js to initialize
   */
  async waitForThreeJS(timeout = 10000) {
    await this.page.waitForFunction(() => {
      return window.THREE !== undefined && document.querySelector('canvas') !== null;
    }, { timeout });
  }

  /**
   * Wait for AR components to load
   */
  async waitForARComponents() {
    await this.page.waitForSelector('[data-testid="3d-preview"]', { timeout: 10000 });
    await this.waitForThreeJS();
  }

  /**
   * Simulate AR session start
   */
  async startARSession() {
    await this.mockWebXRSupport(true);
    const arButton = this.page.locator('button[data-testid="enter-ar"]');
    if (await arButton.isVisible()) {
      await arButton.click();
    }
  }

  /**
   * Check AR status
   */
  async getARStatus() {
    const statusElement = this.page.locator('[data-testid="ar-status"]');
    if (await statusElement.isVisible()) {
      return await statusElement.textContent();
    }
    return 'Unknown';
  }

  /**
   * Simulate camera permissions
   */
  async grantCameraPermissions() {
    try {
      await this.page.context().grantPermissions(['camera']);
    } catch (error) {
      console.log('Camera permissions not available in test environment');
    }
  }

  /**
   * Mock device capabilities
   */
  async mockDeviceCapabilities(capabilities = {}) {
    const defaultCapabilities = {
      isMobile: false,
      hasAccelerometer: false,
      hasGyroscope: false,
      hasCamera: true,
      supportsWebXR: true
    };

    const deviceCaps = { ...defaultCapabilities, ...capabilities };

    await this.page.addInitScript((caps) => {
      // Mock device detection
      window.testDeviceCapabilities = caps;
      
      // Mock mobile detection
      if (caps.isMobile) {
        Object.defineProperty(navigator, 'userAgent', {
          writable: true,
          value: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
        });
      }

      // Mock sensor APIs
      if (!caps.hasAccelerometer) {
        delete window.DeviceMotionEvent;
      }
      if (!caps.hasGyroscope) {
        delete window.DeviceOrientationEvent;
      }
    }, deviceCaps);
  }

  /**
   * Interact with 3D canvas
   */
  async interact3D(action = 'click', position = { x: 200, y: 200 }) {
    const canvas = this.page.locator('canvas');
    await canvas.waitFor({ state: 'visible' });

    switch (action) {
      case 'click':
        await canvas.click({ position });
        break;
      case 'hover':
        await canvas.hover({ position });
        break;
      case 'drag':
        await canvas.dragTo(canvas, {
          sourcePosition: position,
          targetPosition: { x: position.x + 50, y: position.y + 50 }
        });
        break;
      case 'zoom':
        await canvas.wheel(0, -100);
        break;
    }
  }

  /**
   * Check for error notifications
   */
  async checkForErrors() {
    const errorSelectors = [
      '.error-notification',
      '.alert-error',
      '[role="alert"]',
      '.text-red-500'
    ];

    for (const selector of errorSelectors) {
      const errorElement = this.page.locator(selector);
      if (await errorElement.isVisible()) {
        return await errorElement.textContent();
      }
    }
    return null;
  }

  /**
   * Simulate network conditions
   */
  async simulateNetworkConditions(conditions = 'slow') {
    const presets = {
      slow: { downloadThroughput: 50 * 1024, uploadThroughput: 20 * 1024, latency: 500 },
      fast: { downloadThroughput: 10 * 1024 * 1024, uploadThroughput: 5 * 1024 * 1024, latency: 20 },
      offline: { offline: true }
    };

    const networkCondition = presets[conditions] || conditions;
    
    try {
      await this.page.context().setOffline(networkCondition.offline || false);
      if (!networkCondition.offline) {
        // Note: This requires CDP which may not be available in all test environments
        await this.page.context().route('**/*', route => {
          // Add artificial delay
          setTimeout(() => route.continue(), networkCondition.latency || 0);
        });
      }
    } catch (error) {
      console.log('Network condition simulation not supported');
    }
  }

  /**
   * Validate AR accessibility
   */
  async checkARAccessibility() {
    const accessibilityIssues = [];

    // Check for alt text on images
    const images = await this.page.locator('img').all();
    for (const img of images) {
      const alt = await img.getAttribute('alt');
      if (!alt) {
        accessibilityIssues.push('Missing alt text on image');
      }
    }

    // Check for aria-labels on buttons
    const buttons = await this.page.locator('button').all();
    for (const button of buttons) {
      const ariaLabel = await button.getAttribute('aria-label');
      const text = await button.textContent();
      if (!ariaLabel && !text?.trim()) {
        accessibilityIssues.push('Button missing accessible name');
      }
    }

    // Check for heading hierarchy
    const headings = await this.page.locator('h1, h2, h3, h4, h5, h6').all();
    let previousLevel = 0;
    for (const heading of headings) {
      const tagName = await heading.evaluate(el => el.tagName);
      const currentLevel = parseInt(tagName.charAt(1));
      if (currentLevel > previousLevel + 1) {
        accessibilityIssues.push(`Heading level jump: ${tagName} after h${previousLevel}`);
      }
      previousLevel = currentLevel;
    }

    return accessibilityIssues;
  }

  /**
   * Performance monitoring
   */
  async measurePerformance() {
    return await this.page.evaluate(() => {
      const perfData = {
        memory: performance.memory ? {
          used: performance.memory.usedJSHeapSize,
          total: performance.memory.totalJSHeapSize,
          limit: performance.memory.jsHeapSizeLimit
        } : null,
        navigation: performance.getEntriesByType('navigation')[0],
        paint: performance.getEntriesByType('paint'),
        fps: null
      };

      // Rough FPS calculation
      let frames = 0;
      let lastTime = performance.now();
      
      const measureFPS = () => {
        frames++;
        const currentTime = performance.now();
        if (currentTime >= lastTime + 1000) {
          perfData.fps = Math.round(frames * 1000 / (currentTime - lastTime));
          return perfData;
        }
        requestAnimationFrame(measureFPS);
      };

      requestAnimationFrame(measureFPS);
      
      return perfData;
    });
  }
}

/**
 * Pattern generation utilities
 */
export class PatternTestHelpers {
  static generateComplexPattern() {
    return {
      svgContent: `
        <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="dots" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
              <circle cx="10" cy="10" r="2" fill="#8B5CF6"/>
            </pattern>
          </defs>
          <rect width="300" height="300" fill="url(#dots)" opacity="0.3"/>
          <path d="M50,150 Q150,50 250,150 Q150,250 50,150 Z" fill="none" stroke="#8B5CF6" stroke-width="3"/>
          <circle cx="150" cy="150" r="80" fill="none" stroke="#8B5CF6" stroke-width="2"/>
          <path d="M70,70 L230,230 M230,70 L70,230" stroke="#8B5CF6" stroke-width="1"/>
        </svg>
      `,
      patternId: 'complex-' + Date.now(),
      name: 'Complex Test Pattern'
    };
  }

  static generateSimplePattern() {
    return {
      svgContent: `
        <svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
          <circle cx="50" cy="50" r="40" fill="none" stroke="#8B5CF6" stroke-width="2"/>
        </svg>
      `,
      patternId: 'simple-' + Date.now(),
      name: 'Simple Test Pattern'
    };
  }
}

/**
 * Test data constants
 */
export const TEST_CONSTANTS = {
  TIMEOUTS: {
    SHORT: 2000,
    MEDIUM: 5000,
    LONG: 10000,
    VERY_LONG: 30000
  },
  
  VIEWPORTS: {
    MOBILE_PORTRAIT: { width: 375, height: 667 },
    MOBILE_LANDSCAPE: { width: 667, height: 375 },
    TABLET: { width: 768, height: 1024 },
    DESKTOP: { width: 1920, height: 1080 }
  },
  
  SELECTORS: {
    AR_BUTTON: 'button:has-text("Launch AR")',
    AR_STATUS: '[data-testid="ar-status"]',
    THREE_CANVAS: 'canvas',
    PATTERN_CARD: '[data-testid="pattern-card"]',
    ERROR_NOTIFICATION: '.error-notification'
  }
};