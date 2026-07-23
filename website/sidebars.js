/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    {
      type: 'doc',
      id: 'index',
      label: 'Overview',
    },
    {
      type: 'category',
      label: 'Concepts',
      collapsed: false,
      items: [
        'concepts/framework',
        'concepts/cases',
        'concepts/diagnose_correct',
      ],
    },
    {
      type: 'category',
      label: 'Tutorial',
      collapsed: false,
      items: [
        'tutorial/quickstart',
        'tutorial/case1',
        'tutorial/case2',
        'tutorial/case3',
        'tutorial/case4',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      collapsed: false,
      items: [
        'api/validator',
        'api/checks',
        'api/adjustments',
        'api/data_handler',
        'api/planner',
      ],
    },
  ],
};

module.exports = sidebars;
