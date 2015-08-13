Quick Start
===========

This requires [torch](http://torch.ch/docs/getting-started.html). Once
installed, run `th logic.th`. You should see output that looks
something like:

```
Initial Bill	Term(-0.049130, -0.285756, 0.425711, 0.144995, -0.340230)
Initial Mary	Term(0.045865, 0.041869, -0.210560, -0.219520, -0.264111)
Initial run	Term(-0.305898, -0.128068, -0.446399, -0.414933, 0.287786)
Iteration:	1		Sentence:  1	      grad:	 1	
Iteration:	1		Sentence:  2	      grad:	 1	
Iteration:	1		Sentence:  3	      grad:	 -0.1	
Iteration:	2		Sentence:  1	      grad:	 1	
Iteration:	2		Sentence:  3	      grad:	 -0.1	
Iteration:	2		Sentence:  2	      grad:	 1	
Iteration:	3		Sentence:  3	      grad:	 -0.1	
Iteration:	3		Sentence:  2	      grad:	 -0.1	
Iteration:	3		Sentence:  1	      grad:	 1	
Iteration:	4		Sentence:  2	      grad:	 -0.1	
Iteration:	4		Sentence:  1	      grad:	 -0.1	
Iteration:	4		Sentence:  3	      grad:	 1	
Iteration:	5		Sentence:  1	      grad:	 1	
Iteration:	5		Sentence:  3	      grad:	 1	
Iteration:	5		Sentence:  2	      grad:	 -0.1	
Iteration:	6		Sentence:  2	      grad:	 -0.1	
Iteration:	6		Sentence:  3	      grad:	 -0.1	
Iteration:	6		Sentence:  1	      grad:	 1	
Iteration:	7		Sentence:  1	      grad:	 -0.1	
Iteration:	7		Sentence:  2	      grad:	 -0.1	
Iteration:	7		Sentence:  3	      grad:	 -0.1	
Bill		Term(-0.009130, -0.005756, 0.425711, 0.144995, -0.340230)
Mary		Term(0.045865, 0.001869, -0.210560, -0.219520, -0.264111)
Run		Term(-0.305898, 0.071932, -0.446399, -0.414933, 0.287786)
```


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
