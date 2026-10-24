// Add header links back to the site and to the same page in the other
// language. Both books share chapter paths, so the page path carries over.
(function () {
  var buttons = document.querySelector("#mdbook-menu-bar .right-buttons");
  if (!buttons) {
    return;
  }
  var ja = document.documentElement.lang === "ja";
  var root = ja ? "/ja/docs/rhai/" : "/docs/rhai/";
  var deployed = location.pathname.indexOf(root) === 0;
  var site = deployed ? "" : "https://www.conchordal.org";

  function link(text, href, lang) {
    var a = document.createElement("a");
    a.className = "site-nav-link";
    a.textContent = text;
    a.href = href;
    if (lang) {
      a.lang = lang;
      a.hreflang = lang;
    }
    buttons.insertBefore(a, buttons.firstChild);
  }

  if (deployed) {
    var page = location.pathname.slice(root.length);
    link(ja ? "English" : "日本語", (ja ? "/docs/rhai/" : "/ja/docs/rhai/") + page + location.hash, ja ? "en" : "ja");
  }
  link("conchordal.org", site + (ja ? "/ja/" : "/"));

  var style = document.createElement("style");
  style.textContent =
    ".site-nav-link{color:var(--icons);font-size:1.5rem;margin-inline-end:1.2rem;text-decoration:none;white-space:nowrap}" +
    ".site-nav-link:hover{color:var(--icons-hover)}";
  document.head.appendChild(style);
})();
