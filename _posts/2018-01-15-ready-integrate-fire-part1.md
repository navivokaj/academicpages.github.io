---
title: "'Ready, Integrate, Fire!' Part 1: Solving a simple neuronal model"
date: 2018-01-15
permalink: /posts/2018/01/ready-integrate-fire-part1/
tags:
  - computational neuroscience
  - neural computation
  - integrate-and-fire model
---

*For my first blog post, I will find the analytical solution of a leaky integrate-and-fire (LIF) model of a neuron under a constant injected current. Too much to handle? Read more to find out! (This is Part 1 of a series of four posts on LIF model)*

My love for numbers has sparked my interest on how I can use mathematics to model the complex processes happening inside the human brain. So when I created this blog, I wanted to start with modelling just a single neuron—the building block of our brains. The model I chose to focus on comes in an intriguing (and somehow provocative!) name: leaky integrate-and-fire (LIF) model.

Why did I choose LIF model? It’s a simple model. That’s it. In my journey to becoming a computational neuroscientist, I have to start with understanding the basic models in the field of neuroscience. Despite its ‘simplicity’, I delved a lot into this model, and I even think that it deserves more spotlight that I decided to dedicate my first four blog posts on this single model! For my first blog post, I will be finding the analytical solution of a leaky integrate-and-fire (LIF) model of a neuron under a static input current. Ready to know more? Let’s start! 

## A first look at integrate-and-fire models

The integrate-and-fire model was first proposed by the French neuroscientist Louis Lapicque in 1907. It is probably one of the simplest and most commonly used mathematical models describing the basic electrical properties of a single neuron. (We’ll deal with the ‘leaky’ term later.) These properties are as follows:

[1] A neuron will typically fire an action potential when its membrane potential reaches a critical threshold, which is about -55 to -50 millivolts [mV].

[2] During the action potential, the membrane potential follows a quick, high trajectory and then immediately returns to a value that is hyperpolarized relative to the threshold potential.

The integrate-and-fire model captures all these properties by carrying out the following golden rule:

*Whenever the membrane potential of the model neuron reaches a threshold value $$V_{th}$$, the neuron fires an action potential. After the action potential, the membrane potential is reset to a value $$V_{reset}$$ below the threshold potential, $$ V_{reset} &lt; V_{th}.$$*

The above rule spells out how the model got its name: as the membrane potential accumulates (or integrates in continuous terms, much like what an integral does) up to the point-of-no-return (the threshold value), the neuron releases (aka fires) an action potential. Ready, integrate, fire!

## The model’s strengths and weaknesses

Of course, having a simple model means that one must forego incorporating complex properties that would make the neuronal model more realistic. The integrate-and-fire model comes with the limitation of not including the biophysical mechanisms responsible for action potentials, such as ion channel kinetics. Despite this, the model is still an extremely useful one as it focuses on modelling the subthreshold potential dynamics. As stated in [this paper created by the Goldman Lab at UC Davis](http://neuroscience.ucdavis.edu/goldman/Tutorials_files/Integrate&Fire.pdf), another reason why we can let go of the biophysical properties in modelling a neuron is that we are not too much interested with the exact shape of an action potential, and the only informative feature we can probably get from a neuron’s spiking is the exact times at which action potentials occur.

## So what’s up with a ‘leaky’ neuron?

As previously mentioned, integrate-and-fire models are concerned with subthreshold membrane potential dynamics, and this can be done with various levels of modelling rigor. The simplest version of these models is called the leaky (or passive) integrate-and-fire model. Under this LIF model, all active membrane conductances are ignored, including, for the moment, synaptic inputs. Moreover, the entire membrane conductance is modelled as a single passive leakage term, making the model neuron ‘leaky’.  For small fluctuations about the resting membrane potential, neuronal conductances are approximately constant, and the LIF model assumes that this constancy holds over the entire subthreshold range. However, this assumption is reasonable only for some neurons.

## The Leaky Integrate-and-Fire Model

We are now ready to formally state the LIF model. Consider a neuron modelled as a leaky capacitor with membrane resistance $$R_m$$, time constant $$\tau_m = R_mC_m$$ (where $$C_m$$ is the membrane’s capacitance), and resting potential $$E_L$$. Below the threshold value $$V_{th}$$, the equation for the voltage $$V(t)$$ of this neuron when it receives a current injection $$I_e$$ is given by:

$$ \tau_m \dfrac{dV}{dt} = E_L - V + R_mI_e \text{ (Eq 1)} $$ 

To generate action potentials, the above equation is augmented by the golden rule stated above: whenever $$V$$ reaches the threshold value $$V_{th}$$, an action potential is fired and the potential is reset to $$V_{reset}$$.

A thorough derivation of the equation and explanation of its terms will be provided in Part 3 of this LIF series. For now, our concern is on finding the analytical solution of the leaky integrate-and-fire model in response to a constant injected current. 

## Solving the LIF Model

Suppose we know the value of a neuron’s membrane potential at some reference point $$t_0$$. We denote this value as $$V(t_0)$$. We can then compute the analytical solution of the LIF model, which is just a first-order nonhomogeneous differential equation.

We can use the [method of undetermined coefficients](http://www.math.poly.edu/courses/ma1112/ODE1.pdf) to solve this ODE. We first find a solution $$V_h(t)$$ for the homogeneous counterpart of the model. Rearranging terms in Eq 1, we have

$$\tau_m \dfrac{dV}{dt} + V = E_L + R_mI_e \text{ (Eq 2)}$$  

Since the input current $$I_e$$ is assumed to be constant, the right-hand side of the equation also turns out to be a constant, i.e. $$\tau_m \frac{dV}{dt} + V = f(t),$$ where $$f(t)=E_L + R_mI_e$$. By setting $$f(t)=0$$, the homogeneous counterpart is given by

$$\begin{align} \tau_m V’ + V &amp;= 0 \\\\ V’ &amp;= -\dfrac{V}{\tau_m}. \end{align}$$

By separation of variables (or just intuition), the solution to the above equation is simply 

$$V_h (t) = A e^{-t/\tau_m},$$ 

where $$A$$ is some constant.

Next, we need to find a particular solution for Eq 1. Since the right-hand side of Eq 2 is just a constant, we can stipulate that the particular solution is of the form $$V_p (t) = c$$, where $$c$$ is a constant. We can find the value of this constant by plugging the particular solution to Eq 2. That is,

$$\begin{align} \tau_m (c’) + c &= E_L + R_mI_e \\ 0 + c &= E_L + R_mI_e, \end{align}$$

and hence the particular solution is 

$$V_p (t) = E_L + R_mI_e.$$

The general solution for the nonhomogeneous differential equation Eq 1 is expressed as the sum of the homogeneous solution $$V_h (t)$$ and the particular solution $$V_p (t)$$. That is,

$$\begin{align} V(t) &= V_h (t) + V_p (t) \\ V(t) &= A e^{-t/\tau_m} + E_L + R_mI_e \text{ (Eq 3)} \end{align}$$ 

In order to find the exact value of the constant $$A$$ in the above equation, we simply use the initial condition that we know the value $$V(t_0)$$ of the neuron’s membrane potential at the reference point $$t= t_0$$:

$$\begin{align} V(t_0) &= A e^{-t_0/\tau_m} + E_L + R_mI_e \\ A &= [V(t_0) - E_L -R_mI_e]e^{t_0/\tau_m}. \end{align}$$

Substituting this value in Eq 3 and rearranging the terms, we have

$$V(t) = E_L + R_mI_e + [V(t_0) - E_L -R_mI_e]e^{-(t-t_0)/\tau_m}, \text{ (Eq 4)}$$ 

which gives us the analytical solution to a leaky integrate-and-fire model when a constant current is injected to the neuron. We note that this solution is valid for the LIF model only as long as the membrane potential $$V$$ stays below the threshold $$V_{th}$$.

For the math geeks out there, we could also use the Laplace transform to calculate the exact solution of the LIF model. In fact, I first used Laplace to solve it, but I decided to present the neater and simpler solution here.

## Calculating the theoretical firing rate

We can also compute the theoretical firing rate of a leaky integrate-and-fire model neuron in response to a static input current. The firing rate of a neuron is defined as the number of spikes it fires per second [measured in Hz]. We can calculate it by determining the interspike interval $$t_{isi}$$, which is the amount of time between two consecutive spikes. The interspike-interval firing rate is then simply 1 over $$t_{isi}$$, indicating that the LIF model fires once over the interspike interval.

Suppose that at reference point $$t = t_0$$, the neuron has just fired an action potential and is thus reset to the value $$V_{reset}$$, i.e. $$V(t_0) = V_{reset}$$. The next spike will then occur after $$t_{isi}$$ unit of time, that is, at time $$t=t_0 + t_{isi}$$. At this time, the membrane potential reaches the threshold $$V_{th}$$, and thus we have

$$V(t_0 + t_{isi}) = V_{th} = E_L + R_mI_e + [V_{reset} - E_L -R_mI_e]e^{-t_{isi}/\tau_m}.$$

Solving for $$t_{isi}$$ from the above equation gives us

$$t_{isi}=\tau_m\ln \Big( \dfrac{R_mI_e + E_L - V_{reset}}{R_mI_e + E_L - V_{th}} \Big).$$

Therefore, the theoretical interspike-interval firing rate $$r_{isi}$$ is given by the inverse of $$t_{isi}$$:

$$r_{isi} = \dfrac{1}{t_{isi}} = \Big[ \tau_m\ln \Big( \dfrac{R_mI_e + E_L - V_{reset}}{R_mI_e + E_L - V_{th}} \Big) \Big] ^{-1}. \text{ (Eq 5)}$$

We note that the above firing rate only exists for some range of values of the injected current $$I_e$$. We can get the domain of $$r_{isi}$$ by solving the following inequality:

$$\dfrac{R_mI_e + E_L - V_{reset}}{R_mI_e + E_L - V_{th}} > 0$$

This branches into two cases:

*Case 1:* both numerator and denominator are positive 

$$R_mI_e + E_L - V_{reset} > 0 \text{ and } R_mI_e + E_L - V_{th} > 0$$

$$R_mI_e + E_L > V_{reset} \text{ and } R_mI_e + E_L > V_{th}.$$

But since $$V_{th} > V_{reset}$$, the above inequality reduces to

$$\begin{align} R_mI_e + E_L &> V_{th}, \text{ or} \\ I_e &> \dfrac{V_{th}-E_L}{R_m}. \end{align} $$

*Case 2:* both numerator and denominator are negative 

$$R_mI_e + E_L - V_{reset} < 0 \text{ and } R_mI_e + E_L - V_{th} < 0$$

$$R_mI_e + E_L < V_{reset} \text{ and } R_mI_e + E_L < V_{th}.$$

But since $$V_{reset} < V_{th}$$, the above condition reduces to

$$\begin{align} R_mI_e + E_L &< V_{reset}, \text{ or} \\ I_e &< \dfrac{V_{reset}-E_L}{R_m}. \end{align} $$

We have to note that there is no negative membrane resistance and that $$V_{reset}$$  is always less than or equal to the resting potential $$E_L$$, making the right-hand side of the above inequality to be nonpositive. However, the injected current $$I_e$$ is always nonnegative, making the above inequality to be invalid.

This therefore leads us to only consider the result of the first case. If we set the threshold current as $$I_{th} = \frac{V_{th}-E_L}{R_m}$$, then $$r_{isi}$$ is defined by Eq 5 if the injected current $$I_e$$ is greater than $$I_{th}$$. Otherwise, the neuron will not fire and thus $$r_{isi} = 0$$.Thus, the theoretical interspike-interval firing rate $$r_{isi}$$ is formally given by

$$\begin{align}
r_{isi} =
\begin{cases}
 \Big[ \tau_m\ln \Big( \dfrac{R_mI_e + E_L - V_{reset}}{R_mI_e + E_L - V_{th}} \Big)\Big] ^{-1} &\text{ if } I_e > I_{th} = \dfrac{V_{th}-E_L}{R_m}  \\
  0 &\text{ if } I_e \leq I_{th}.
 \end{cases}
 \end{align}$$

So that concludes my first blog post. On the second one in this four-part series, I will simulate the leaky integrate-and-fire model using Python and verify the theoretical interspike-interval firing rate we just computed. ’Til next time, brain folks!

## References

[1] Theoretical Neuroscience by Peter Dayan and L.F. Abbott

[2] Integrate-and-Fire Model Tutorial by the Goldman Lab at UC Davis ([link](http://neuroscience.ucdavis.edu/goldman/Tutorials_files/Integrate&Fire.pdf))

[3] Computational Neuroscience course offered by University of Washington on Coursera [which I just completed! YAY! I’ll write more about my experience on another blog post :)](https://www.coursera.org/account/accomplishments/records/L3L4JS3U6BDW)
