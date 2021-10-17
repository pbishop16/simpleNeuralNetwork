module.exports = {
  env: {
    node: true,
    es2021: true,
  },
  extends: ["prettier"],
  plugins: ["prettier"],
  parserOptions: {
    ecmaVersion: 12,
    sourceType: "module",
  },
  rules: {
    "prettier/prettier": "error",
    indent: ["error", 2],
    "linebreak-style": ["error", "unix"],
    quotes: ["error", "single"],
    semi: ["error", "always"],
  },
};
