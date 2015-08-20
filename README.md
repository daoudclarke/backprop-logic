Quick Start
===========

This requires [torch](http://torch.ch/docs/getting-started.html). Once
installed, run `th logic.th`. You should see output that looks
something like:

```
Initial Bill	Term(-0.020758, 0.297449, 0.137401, -0.057877, -0.050278)
Initial Mary	Term(0.443954, 0.325557, 0.005245, -0.380538, 0.273569)
Initial run	Term(-0.152070, -0.034093, 0.310034, -0.044150, -0.113796)
Iteration:	1		Sentence:  1	     grad:	0	
Iteration:	1		Sentence:  3	     grad:	1	
Iteration:	1		Sentence:  2	     grad:	0	
Iteration:	2		Sentence:  1	     grad:	0	
Iteration:	2		Sentence:  3	     grad:	1	
Iteration:	2		Sentence:  2	     grad:	0	
Iteration:	3		Sentence:  1	     grad:	0	
Iteration:	3		Sentence:  3	     grad:	1	
Iteration:	3		Sentence:  2	     grad:	0	
Iteration:	4		Sentence:  3	     grad:	1	
Iteration:	4		Sentence:  1	     grad:	0	
Iteration:	4		Sentence:  2	     grad:	1	
Bill		Term(-0.020758, -0.002551, 0.137401, -0.057877, -0.050278)
Mary		Term(0.443954, 0.325557, -0.094755, -0.380538, 0.273569)
Run		Term(-0.152070, 0.065907, 0.310034, -0.044150, -0.113796)
```

What's happening here?
----------------------

We are using stochastic gradient descent and backpropagation to find a
model for three logical sentences, which can be roughly expressed as:

 - Bill runs
 - Mary runs
 - Bill and Mary are different people

These three sentences are currently hard coded in the `logic.lua`
file. The model is expressed as three vectors which we interpret as
predicates, one for each of `bill`, `mary` and `run`.

We can view each dimension as corresponding to a different entity. If
that dimension is positive for a predicate, then we consider it
"true". For example, initially the second and third entities are
`bill`. The third one is also `mary` which makes the third sentence
false. We can see that the gradient for the third sentence is 1 on
each iteration, until finally this sentence is true: in the end
configuration, only the third entity is `bill`, while the first,
second and fifth entities are `mary`, and the second and third
entities have `run` as true.

Backpropagation for Model Building
==================================

There's been a lot of interest recently in combining natural language
semantics with neural networks. One example of this is the excitement
over memory networks, as in
[this paper](http://www.aclweb.org/anthology/P/P15/P15-1150.pdf) at
ACL 2015 by Tai, Socher and Manning.

Here I want to talk about something slightly different, although the
goal is similar, namely to reason about natural language
semantics. The idea is that we can use the same technique used to
train neural networks, backpropagation, to do something completely
different: model building. Model building solves the converse problem
to
[automated theorem proving](https://en.wikipedia.org/wiki/Automated_theorem_proving).
A model builder is useful when you want to know if a logical statement
is [satisfiable](https://en.wikipedia.org/wiki/Satisfiability), but
they have also been used to solve problems in natural language
processing, such as
[textual entailment](http://www.let.rug.nl/bos/pubs/BosMarkert2006MLCW.pdf).

These two things don't immediately seem compatible, as backpropagation
relies on continuous vector space representations, whereas logic is
typically binary. The solution, of course, is to consider nonstandard
logics, such as fuzzy logics which can be interpreted in terms of real
numbers. The idea of combining fuzzy logic and neural networks
[is an old one](https://en.wikipedia.org/wiki/Neuro-fuzzy), but we're
doing something slightly different here.
