// Register a minimal Rhai grammar with the highlight.js instance bundled by
// mdBook, then re-highlight the `rhai` code blocks (mdBook has already run
// auto-detection on them by the time this script executes).
(function () {
  if (typeof hljs === "undefined" || !hljs.registerLanguage) {
    return;
  }
  hljs.registerLanguage("rhai", function (hljs) {
    return {
      name: "Rhai",
      keywords: {
        keyword:
          "fn let const if else switch for in while loop do until " +
          "return break continue throw try catch import export as " +
          "private global this",
        literal: "true false",
      },
      contains: [
        hljs.QUOTE_STRING_MODE,
        hljs.C_LINE_COMMENT_MODE,
        hljs.C_BLOCK_COMMENT_MODE,
        hljs.C_NUMBER_MODE,
      ],
    };
  });
  Array.prototype.forEach.call(
    document.querySelectorAll("code.language-rhai"),
    function (block) {
      block.removeAttribute("data-highlighted");
      if (hljs.highlightElement) {
        hljs.highlightElement(block);
      } else if (hljs.highlightBlock) {
        hljs.highlightBlock(block);
      }
    }
  );
})();
