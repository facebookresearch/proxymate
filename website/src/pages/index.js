/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

const levels = [
  {
    title: 'Representativity',
    question: 'Should the validation data be weighted?',
    content:
      'Detect covariate imbalance and decide whether importance weighting can correct it before anything downstream is computed.',
  },
  {
    title: 'Unit',
    question: 'Does the proxy track the primary per individual?',
    content:
      'Assess accuracy, precision, agreement, calibration, discrimination, and distributional fidelity at the observation level.',
  },
  {
    title: 'Estimate',
    question: 'Does the proxy give the same aggregate answer?',
    content:
      'Check aggregate bias, CI coverage, sign agreement, and effect-size agreement within a single domain.',
  },
  {
    title: 'Domain',
    question: 'Is the proxy–primary relationship stable across domains?',
    content:
      'Use meta-regression across historical domains to assess association, sensitivity, prediction accuracy, and heterogeneity.',
  },
];

const Level = ({title, question, content}) => (
  <div className="col feature-card">
    <div className="feature-card__body">
      <h3>{title}</h3>
      <p className="feature-card__question">{question}</p>
      <p>{content}</p>
    </div>
  </div>
);

const Home = () => {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <div className="hero hero--compact">
        <div className="container text--center">
          <h1 className="hero__title hero__title--wordmark">proxymate</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <p className="hero__tagline">
            A valid inference on a proxy does not guarantee a valid inference on
            the primary. proxymate diagnoses where a proxy breaks down — then
            applies the targeted correction, only when it helps.
          </p>
          <div className="hero__cta">
            <Link
              to="/docs/tutorial/quickstart"
              className="button button--lg button--primary">
              Get Started
            </Link>
            <Link
              to="/docs/"
              className="button button--lg button--ghost">
              Overview
            </Link>
            <Link
              to="/docs/concepts/framework"
              className="button button--lg button--ghost">
              The Framework
            </Link>
          </div>
        </div>
      </div>

      <div className="section section--features" id="levels">
        <div className="container">
          <div className="section__header text--center">
            <h2>Four Levels of Validation</h2>
            <p>
              Each level has its own diagnostic checks, default thresholds, and
              targeted adjustments. Corrections feed forward from one level to
              the next.
            </p>
          </div>
          <div className="row">
            {levels.map((level) => (
              <Level key={level.title} {...level} />
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Home;
