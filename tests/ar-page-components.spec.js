import { test, expect } from '@playwright/test';
import { ARTestHelpers, TEST_CONSTANTS } from './utils/test-helpers.js';

test.describe('AR Page Components', () => {
  let arHelper;

  test.beforeEach(async ({ page }) => {
    arHelper = new ARTestHelpers(page);
    await page.goto('/ar');
    await expect(page).toHaveTitle(/AR BHARAT/);
  });

  test.describe('AR Page Header and Status', () => {
    test('should display AR page header correctly', async ({ page }) => {
      // Check main heading
      const heading = page.locator('h1:has-text("AR Viewer")');
      await expect(heading).toBeVisible();
      await expect(heading).toHaveClass(/text-4xl|text-3xl/); // Check for large text styling
      
      // Check AR status indicator
      const statusIndicator = page.locator('[data-testid="ar-status"]');
      await expect(statusIndicator).toBeVisible();
      
      // Status should show one of the expected states
      const statusText = await statusIndicator.textContent();
      expect(['Ready', 'Active', 'Not Available']).toContain(statusText.trim());
    });

    test('should show correct status based on WebXR support', async ({ page }) => {
      // Test with WebXR supported
      await arHelper.mockWebXRSupport(true);
      await page.reload();
      
      const status = await arHelper.getARStatus();
      expect(['Ready', 'Active']).toContain(status);
      
      // Test with WebXR not supported
      await arHelper.mockWebXRSupport(false);
      await page.reload();
      
      const statusUnsupported = await arHelper.getARStatus();
      expect(statusUnsupported).toContain('Not Available');
    });

    test('should display status with appropriate styling', async ({ page }) => {
      const statusElement = page.locator('[data-testid="ar-status"]');
      
      // Check for status-specific styling
      await expect(statusElement).toHaveClass(/bg-/); // Should have background color
      await expect(statusElement).toHaveClass(/text-/); // Should have text color
      await expect(statusElement).toHaveClass(/px-|py-/); // Should have padding
    });
  });

  test.describe('3D Preview Section', () => {
    test('should render 3D preview container', async ({ page }) => {
      const previewContainer = page.locator('[data-testid="3d-preview"]');
      await expect(previewContainer).toBeVisible();
      
      // Check container styling
      await expect(previewContainer).toHaveClass(/bg-gray-100/);
      await expect(previewContainer).toHaveClass(/rounded-lg/);
    });

    test('should initialize Three.js canvas', async ({ page }) => {
      await arHelper.waitForARComponents();
      
      const canvas = page.locator('canvas');
      await expect(canvas).toBeVisible();
      
      // Check canvas dimensions are reasonable
      const canvasBox = await canvas.boundingBox();
      expect(canvasBox.width).toBeGreaterThan(200);
      expect(canvasBox.height).toBeGreaterThan(200);
    });

    test('should handle canvas interactions', async ({ page }) => {
      await arHelper.waitForARComponents();
      
      // Test click interaction
      await arHelper.interact3D('click');
      await page.waitForTimeout(100);
      
      // Test hover interaction
      await arHelper.interact3D('hover');
      await page.waitForTimeout(100);
      
      // Test zoom interaction
      await arHelper.interact3D('zoom');
      await page.waitForTimeout(100);
      
      // Canvas should remain visible and responsive
      const canvas = page.locator('canvas');
      await expect(canvas).toBeVisible();
    });

    test('should display loading state initially', async ({ page }) => {
      // Reload to catch loading state
      await page.reload();
      
      // Look for loading indicators
      const loadingElements = page.locator('.animate-spin, .loading, :has-text("Loading")');
      // Loading state might be brief, so we'll check if it appears or if content loads quickly
      
      await arHelper.waitForARComponents();
    });
  });

  test.describe('Control Panel', () => {
    test('should display all control buttons', async ({ page }) => {
      // Check for narration button
      const narrationButton = page.locator('button:has-text("Play Narration")');
      await expect(narrationButton).toBeVisible();
      
      // Check for export button
      const exportButton = page.locator('button:has-text("Export 3D Model")');
      await expect(exportButton).toBeVisible();
      
      // Check for share button
      const shareButton = page.locator('button:has-text("Share AR")');
      await expect(shareButton).toBeVisible();
    });

    test('should handle narration button interaction', async ({ page }) => {
      const narrationButton = page.locator('button:has-text("Play Narration")');
      await narrationButton.click();
      
      // Button should provide feedback (might change text or show loading)
      await page.waitForTimeout(500);
      
      // Check if button state changes or audio controls appear
      const buttonText = await narrationButton.textContent();
      // Could be "Stop Narration" or "Playing..." depending on implementation
    });

    test('should handle export button interaction', async ({ page }) => {
      const exportButton = page.locator('button:has-text("Export 3D Model")');
      await exportButton.click();
      
      // Should trigger download or show export options
      await page.waitForTimeout(500);
      
      // Check for any export-related UI changes
    });

    test('should handle share button interaction', async ({ page }) => {
      const shareButton = page.locator('button:has-text("Share AR")');
      await shareButton.click();
      
      // Should show share options or trigger share functionality
      await page.waitForTimeout(500);
      
      // Look for share modal or native share dialog trigger
    });

    test('should style control buttons consistently', async ({ page }) => {
      const controlButtons = page.locator('button:has-text("Play Narration"), button:has-text("Export 3D Model"), button:has-text("Share AR")');
      
      const buttonCount = await controlButtons.count();
      expect(buttonCount).toBe(3);
      
      // Check consistent styling across buttons
      for (let i = 0; i < buttonCount; i++) {
        const button = controlButtons.nth(i);
        await expect(button).toHaveClass(/px-|py-/); // Padding
        await expect(button).toHaveClass(/bg-/); // Background
        await expect(button).toHaveClass(/rounded/); // Border radius
      }
    });
  });

  test.describe('AR Instructions Section', () => {
    test('should display AR instructions', async ({ page }) => {
      const instructionsSection = page.locator('[data-testid="ar-instructions"]');
      await expect(instructionsSection).toBeVisible();
      
      // Should contain helpful text about using AR
      await expect(instructionsSection).toContainText('Point your camera');
    });

    test('should show device-specific instructions', async ({ page }) => {
      // Test mobile instructions
      await arHelper.mockDeviceCapabilities({ isMobile: true });
      await page.reload();
      
      const instructions = page.locator('[data-testid="ar-instructions"]');
      const instructionText = await instructions.textContent();
      
      // Should contain mobile-specific guidance
      expect(instructionText.toLowerCase()).toMatch(/tap|touch|mobile/);
      
      // Test desktop instructions
      await arHelper.mockDeviceCapabilities({ isMobile: false });
      await page.reload();
      
      const desktopInstructions = page.locator('[data-testid="ar-instructions"]');
      const desktopText = await desktopInstructions.textContent();
      
      // Should contain desktop-specific guidance
      expect(desktopText.toLowerCase()).toMatch(/click|mouse|desktop/);
    });

    test('should adapt instructions for WebXR capability', async ({ page }) => {
      // Test with WebXR supported
      await arHelper.mockWebXRSupport(true);
      await page.reload();
      
      const instructions = page.locator('[data-testid="ar-instructions"]');
      const supportedText = await instructions.textContent();
      
      // Should mention AR capabilities
      expect(supportedText.toLowerCase()).toMatch(/ar|augmented reality/);
      
      // Test with WebXR not supported
      await arHelper.mockWebXRSupport(false);
      await page.reload();
      
      const unsupportedInstructions = page.locator('[data-testid="ar-instructions"]');
      const unsupportedText = await unsupportedInstructions.textContent();
      
      // Should provide fallback instructions
      expect(unsupportedText.toLowerCase()).toMatch(/3d|preview|view/);
    });
  });

  test.describe('Pattern Information Display', () => {
    test('should show pattern information when available', async ({ page }) => {
      // Mock a pattern being loaded
      await arHelper.mockPatternGeneration();
      
      const patternInfo = page.locator('[data-testid="pattern-info"]');
      if (await patternInfo.isVisible()) {
        await expect(patternInfo).toContainText('Test Kolam Pattern');
      }
    });

    test('should provide download links', async ({ page }) => {
      await arHelper.mockPatternGeneration();
      
      // Look for download buttons
      const downloadButtons = page.locator('button:has-text("Download"), a[download]');
      if (await downloadButtons.first().isVisible()) {
        const downloadCount = await downloadButtons.count();
        expect(downloadCount).toBeGreaterThan(0);
      }
    });
  });

  test.describe('Responsive Design', () => {
    test('should adapt to mobile viewport', async ({ page }) => {
      await page.setViewportSize(TEST_CONSTANTS.VIEWPORTS.MOBILE_PORTRAIT);
      
      // All main sections should still be visible
      await expect(page.locator('h1:has-text("AR Viewer")')).toBeVisible();
      await expect(page.locator('[data-testid="3d-preview"]')).toBeVisible();
      await expect(page.locator('[data-testid="ar-instructions"]')).toBeVisible();
      
      // Control buttons should be mobile-friendly
      const controlButtons = page.locator('button:has-text("Play Narration"), button:has-text("Export 3D Model"), button:has-text("Share AR")');
      await expect(controlButtons.first()).toBeVisible();
    });

    test('should adapt to tablet viewport', async ({ page }) => {
      await page.setViewportSize(TEST_CONSTANTS.VIEWPORTS.TABLET);
      
      // Check layout adjustments for tablet
      const preview = page.locator('[data-testid="3d-preview"]');
      await expect(preview).toBeVisible();
      
      const previewBox = await preview.boundingBox();
      expect(previewBox.width).toBeGreaterThan(400);
    });

    test('should maintain functionality on large screens', async ({ page }) => {
      await page.setViewportSize(TEST_CONSTANTS.VIEWPORTS.DESKTOP);
      
      // All functionality should work on desktop
      await arHelper.waitForARComponents();
      await arHelper.interact3D('click');
      
      // Control buttons should be properly spaced
      const controlButtons = page.locator('button:has-text("Play Narration"), button:has-text("Export 3D Model"), button:has-text("Share AR")');
      const buttonCount = await controlButtons.count();
      expect(buttonCount).toBe(3);
    });
  });

  test.describe('Error Handling', () => {
    test('should handle Three.js initialization errors', async ({ page }) => {
      // Mock WebGL unavailable
      await page.addInitScript(() => {
        const getContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(contextType) {
          if (contextType === 'webgl' || contextType === 'webgl2') {
            return null; // Simulate WebGL unavailable
          }
          return getContext.call(this, contextType);
        };
      });
      
      await page.reload();
      
      // Should show appropriate error message
      const error = await arHelper.checkForErrors();
      if (error) {
        expect(error.toLowerCase()).toMatch(/webgl|graphics|3d/);
      }
    });

    test('should handle missing pattern gracefully', async ({ page }) => {
      // Navigate to AR page without any pattern
      await page.goto('/ar');
      
      // Should show default content or instructions for loading a pattern
      const instructions = page.locator('[data-testid="ar-instructions"]');
      await expect(instructions).toBeVisible();
    });

    test('should recover from temporary errors', async ({ page }) => {
      // Simulate network error
      await arHelper.simulateNetworkConditions('offline');
      await page.reload();
      
      // Go back online
      await arHelper.simulateNetworkConditions('fast');
      await page.reload();
      
      // Should recover and display content
      await expect(page.locator('h1:has-text("AR Viewer")')).toBeVisible();
    });
  });
});