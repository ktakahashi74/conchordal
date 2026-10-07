;;; worker-shell.el --- Codex worker shells for the orchestrating session -*- lexical-binding: t; -*-

;; Loaded on demand through `agent-emacsclient'.  Each worker is an ordinary
;; agent-shell buffer, so the author can watch it and type into it.

(require 'agent-shell)
(require 'agent-shell-openai)

(defconst conchordal-worker-root "/home/shafi/lwrk/conchordal/"
  "Checkout the workers start in.")

(defun conchordal-worker-start (name)
  "Start a background Codex worker shell called NAME and return its buffer name.
The model and effort come from the Codex defaults of this Emacs.  The KEIO
account is forced because Codex otherwise resolves this checkout to RIKEN."
  (let* ((default-directory conchordal-worker-root)
         (my/profile-override 'keio)
         (buffer (agent-shell--start
                  :config (agent-shell-openai-make-codex-config)
                  :no-focus t
                  :new-session t
                  :session-strategy 'new)))
    (with-current-buffer buffer
      (rename-buffer (format "Codex worker %s @ conchordal" name) t)
      (buffer-name))))

(defun conchordal-worker-send (buffer-name prompt)
  "Submit PROMPT to the worker shell BUFFER-NAME without selecting it."
  (agent-shell-insert :text prompt
                      :submit t
                      :no-focus t
                      :shell-buffer (get-buffer buffer-name))
  buffer-name)

(defun conchordal-worker-status ()
  "Return (NAME ACCOUNT BUSY) for every worker shell."
  (delq nil
        (mapcar
         (lambda (buffer)
           (with-current-buffer buffer
             (when (string-prefix-p "Codex worker " (buffer-name))
               (list (buffer-name)
                     (bound-and-true-p my/codex-session-profile)
                     (and (bound-and-true-p shell-maker--busy) t)))))
         (agent-shell-buffers))))

(provide 'worker-shell)
;;; worker-shell.el ends here
