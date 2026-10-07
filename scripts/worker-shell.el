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
    ;; A bare `rename-buffer' leaves shell-maker looking up the old name, and
    ;; every later submit fails with (wrong-type-argument processp nil).
    (shell-maker-set-buffer-name buffer (format "Codex worker %s @ conchordal" name))
    (buffer-name buffer)))

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

(defun conchordal-worker-frame (buffer-names &optional columns)
  "Tile BUFFER-NAMES in a frame named \"conchordal workers\", COLUMNS per row.
The frame is created on first use and reused afterwards, so the author's own
frame keeps its windows.  COLUMNS defaults to 4."
  (let* ((columns (or columns 4))
         (orig (selected-frame))
         (frame (or (seq-find (lambda (f)
                                (equal (frame-parameter f 'name) "conchordal workers"))
                              (frame-list))
                    (make-frame '((name . "conchordal workers")
                                  (width . 300) (height . 80)
                                  (fullscreen . maximized)))))
         (rows (ceiling (/ (float (length buffer-names)) columns)))
         (window-min-width 2)
         (window-min-height 1))
    (with-selected-frame frame
      (delete-other-windows)
      (let ((row-windows (list (selected-window)))
            (names buffer-names))
        (dotimes (_ (1- rows))
          (push (split-window (car row-windows) nil 'below) row-windows))
        (dolist (row (nreverse row-windows))
          (let ((w row))
            (dotimes (i columns)
              (when names
                (when (> i 0) (setq w (split-window w nil 'right)))
                (set-window-buffer w (get-buffer (pop names)))))))
        (balance-windows)))
    (select-frame orig)
    (length (window-list frame 'no-minibuf))))

(provide 'worker-shell)
;;; worker-shell.el ends here
