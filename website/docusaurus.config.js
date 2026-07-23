// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// @ts-check
// Note: this config is written as CommonJS so the file is a valid plain `.js`.
// `remark-math` and `rehype-katex` ship as ES modules, so they are pulled in
// via dynamic `import()` from inside an async config factory (a pattern
// Docusaurus supports) rather than a top-level `require`.

/** @returns {Promise<import('@docusaurus/types').Config>} */
module.exports = async function createConfig() {
  const {themes: prismThemes} = await import('prism-react-renderer');
  const remarkMath = (await import('remark-math')).default;
  const rehypeKatex = (await import('rehype-katex')).default;

  return {
    title: 'proxymate',
    tagline:
      'A four-level framework for validating proxy metrics — diagnose, then correct.',

    url: 'https://facebookresearch.github.io',
    baseUrl: '/proxymate/',
    favicon: 'img/logo.svg',

    organizationName: 'facebookresearch',
    projectName: 'proxymate',

    onBrokenLinks: 'throw',
    onBrokenMarkdownLinks: 'warn',

    i18n: {
      defaultLocale: 'en',
      locales: ['en'],
    },

    presets: [
      [
        'classic',
        /** @type {import('@docusaurus/preset-classic').Options} */
        ({
          docs: {
            sidebarPath: './sidebars.js',
            editUrl:
              'https://github.com/facebookresearch/proxymate/tree/main/website/',
            remarkPlugins: [remarkMath],
            rehypePlugins: [rehypeKatex],
          },
          blog: false,
          theme: {
            customCss: './src/css/custom.css',
          },
        }),
      ],
    ],

    stylesheets: [
      {
        href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
        type: 'text/css',
        integrity:
          'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
        crossorigin: 'anonymous',
      },
    ],

    themeConfig:
      /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
      ({
        metadata: [
          {
            name: 'description',
            content:
              'proxymate is a Python framework for validating proxy metrics across four levels — representativity, unit, estimate, and domain — with a diagnose-then-correct workflow.',
          },
          {
            name: 'keywords',
            content:
              'proxy metrics, surrogate endpoints, validation, meta-analysis, PPI, causal inference, Python',
          },
        ],
        navbar: {
          title: 'proxymate',
          items: [
            {
              type: 'docSidebar',
              sidebarId: 'docsSidebar',
              label: 'Docs',
              position: 'left',
            },
            {
              href: 'https://github.com/facebookresearch/proxymate',
              label: 'GitHub',
              position: 'right',
            },
          ],
        },
        footer: {
          style: 'dark',
          copyright: `Licensed under BSD-3-Clause. Copyright © ${new Date().getFullYear()} Meta Platforms, Inc. and affiliates.`,
        },
        prism: {
          theme: prismThemes.github,
          darkTheme: prismThemes.dracula,
        },
      }),
  };
};
