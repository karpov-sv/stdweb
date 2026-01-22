/**
 * Inline Edit - Reusable inline editing component
 *
 * Provides click-to-edit functionality for text fields with auto-expanding textarea.
 *
 * Usage:
 *   InlineEdit.init({
 *     containerSelector: '#title-container',
 *     displaySelector: '#title-display',
 *     textSelector: '#title-text',
 *     addButtonSelector: '#title-add-btn',
 *     editContainerSelector: '#title-edit-container',
 *     inputSelector: '#title-input',
 *     saveButtonSelector: '#title-save-btn',
 *     cancelButtonSelector: '#title-cancel-btn',
 *     updateUrl: '/tasks/123/update_title',
 *     csrfToken: 'token-here',
 *     fieldName: 'title',
 *     onSaveSuccess: function(newValue) { console.log('Saved!'); }
 *   });
 */

var InlineEdit = (function() {
    'use strict';

    function InlineEditComponent(options) {
        // Required selectors
        this.containerSelector = options.containerSelector;
        this.displaySelector = options.displaySelector;
        this.textSelector = options.textSelector;
        this.addButtonSelector = options.addButtonSelector;
        this.editContainerSelector = options.editContainerSelector;
        this.inputSelector = options.inputSelector;
        this.saveButtonSelector = options.saveButtonSelector;
        this.cancelButtonSelector = options.cancelButtonSelector;

        // Configuration
        this.updateUrl = options.updateUrl;
        this.csrfToken = options.csrfToken;
        this.fieldName = options.fieldName || 'value';

        // Callbacks
        this.onSaveSuccess = options.onSaveSuccess || function() {};
        this.onSaveError = options.onSaveError || function(error) { alert('Failed to save: ' + error); };

        // Elements (will be set on init)
        this.$display = null;
        this.$text = null;
        this.$addButton = null;
        this.$editContainer = null;
        this.$input = null;
        this.$saveButton = null;
        this.$cancelButton = null;

        this.originalValue = '';

        this.initialize();
    }

    InlineEditComponent.prototype.initialize = function() {
        var self = this;

        // Find elements
        this.$display = $(this.displaySelector);
        this.$text = $(this.textSelector);
        this.$addButton = $(this.addButtonSelector);
        this.$editContainer = $(this.editContainerSelector);
        this.$input = $(this.inputSelector);
        this.$saveButton = $(this.saveButtonSelector);
        this.$cancelButton = $(this.cancelButtonSelector);

        // Store original value
        this.originalValue = this.$input.val();

        // Bind events
        this.bindEvents();
    };

    InlineEditComponent.prototype.bindEvents = function() {
        var self = this;

        // Show pencil icon on hover
        this.$display.hover(
            function() { $(this).find('.fa-pencil').css('opacity', '0.5'); },
            function() { $(this).find('.fa-pencil').css('opacity', '0'); }
        );

        // Click to edit
        this.$display.on('click', function() { self.enterEdit(); });
        this.$addButton.on('click', function() { self.enterEdit(); });

        // Save and cancel
        this.$saveButton.on('click', function() { self.save(); });
        this.$cancelButton.on('click', function() { self.cancel(); });

        // Auto-resize on input
        this.$input.on('input', function() { self.autoResize(); });

        // Keyboard shortcuts
        this.$input.on('keydown', function(e) {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                self.save();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                self.cancel();
            }
        });
    };

    InlineEditComponent.prototype.autoResize = function() {
        this.$input.css('height', 'auto');
        this.$input.css('height', this.$input[0].scrollHeight + 'px');
    };

    InlineEditComponent.prototype.enterEdit = function() {
        this.$display.hide();
        this.$addButton.hide();
        this.$editContainer.show();
        this.autoResize();
        this.$input.focus();
    };

    InlineEditComponent.prototype.exitEdit = function() {
        this.$editContainer.hide();
        if (this.$input.val().trim()) {
            this.$display.show();
        } else {
            this.$addButton.show();
        }
    };

    InlineEditComponent.prototype.save = function() {
        var self = this;
        var newValue = this.$input.val().trim();

        var postData = {
            csrfmiddlewaretoken: this.csrfToken
        };
        postData[this.fieldName] = newValue;

        $.post({
            url: this.updateUrl,
            data: postData,
            success: function(response) {
                if (response.success) {
                    // Update display text (convert newlines to <br>)
                    self.$text.html(newValue.replace(/\n/g, '<br>'));
                    self.originalValue = newValue;
                    self.exitEdit();

                    // Call success callback
                    self.onSaveSuccess(newValue);
                } else {
                    self.onSaveError(response.error || 'Unknown error');
                }
            },
            error: function(xhr, status, error) {
                self.onSaveError(error || 'Network error');
            }
        });
    };

    InlineEditComponent.prototype.cancel = function() {
        this.$input.val(this.originalValue);
        this.exitEdit();
    };

    // Public API
    return {
        init: function(options) {
            return new InlineEditComponent(options);
        }
    };
})();
