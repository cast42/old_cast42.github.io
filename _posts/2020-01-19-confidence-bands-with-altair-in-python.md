# Plotting regression line with confidence bands in Altair 4

I'm a big fan of [Altair](https://altair-viz.github.io/) created by [Jake Vanderplas](https://github.com/jakevdp). It's a python library that allows for declarative visualisation.
In Version 4, a regression transform is added that makes it easy to draw a regression line on a scatter plot.
The example from the Altair website:

```python
import altair as alt
import pandas as pd
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10)
y = x - 5 + np.random.randn(len(x))

df = pd.DataFrame({'x': x, 'y': y})

chart = alt.Chart(df).mark_point().encode(
    x='x',
    y='y'
)

chart + chart.transform_regression('x', 'y').mark_line()
````

![](/images/altair_regression_transform.png "Result of regression transform on Altair scatter graph")

Even [higher order regression](https://altair-viz.github.io/gallery/poly_fit_regression.html?highlight=regression) lines are possible.

To calculate the confidence band of the regression, I used the Python code from a blogpost titled [the hackers guide to uncertainty estimates](https://erikbern.com/2018/10/08/the-hackers-guide-to-uncertainty-estimates.html) written by [Erik Bernhardsson](https://erikbern.com/)


First we need a helper function that calculates the negative logarithmic likelihood:

```python
import scipy.optimize
import random

def model(xs, k, m):
    return k * xs + m

def neg_log_likelihood(tup, xs, ys):
    # Since sigma > 0, we use use log(sigma) as the parameter instead.
    # That way we have an unconstrained problem.
    k, m, log_sigma = tup
    sigma = np.exp(log_sigma)
    delta = model(xs, k, m) - ys
    return len(xs)/2*np.log(2*np.pi*sigma**2) + \
        np.dot(delta, delta) / (2*sigma**2)
```

Next, we define a function that calculates the upper and lower confidence bands of the regression using bootstrapping.

```python
def confidence_bands(xs, ys, nr_bootstrap):
    curves = []
    xys = list(zip(xs, ys))
    for i in range(nr_bootstrap):
        # sample with replacement
        bootstrap = [random.choice(xys) for _ in xys]
        xs_bootstrap = np.array([x for x, y in bootstrap])
        ys_bootstrap = np.array([y for x, y in bootstrap])
        k_hat, m_hat, log_sigma_hat = scipy.optimize.minimize(
          neg_log_likelihood, (0, 0, 0), args=(xs_bootstrap, ys_bootstrap)
        ).x
        curves.append(
          model(xs, k_hat, m_hat) +
          # Note what's going on here: we're _adding_ the random term
          # to the predictions!
          np.exp(log_sigma_hat) * np.random.normal(size=xs.shape)
        )
    lo, hi = np.percentile(curves, (2.5, 97.5), axis=0)
    return lo, hi
```

Finally, we extend the regression line on the Altair scatter graph with a confidence band:
```python
np.random.seed(42)
xs = np.linspace(0, 10)
ys = xs - 5 + np.random.randn(len(xs))

df = pd.DataFrame({'x':xs, 'y':ys})

df['lo'], df['hi'] = confidence_bands(xs, ys, 1000)

ci = alt.Chart(df).mark_area().encode(
    x=alt.X('x:Q', title=''),
    y=alt.Y('lo:Q', title=''),
    y2=alt.Y2('hi:Q', title=''),
    color=alt.value('lightblue'),
    opacity=alt.value(0.6)
)

chart = alt.Chart(df).mark_point().encode(
    x=alt.X('x', title='x'),
    y=alt.Y('y', title='y')
)

chart + ci + chart.transform_regression('x', 'y').mark_line()
```

![](/images/altair_confidence_band.png "Result of adding confidence band to the regression transform on Altair scatter graph")
