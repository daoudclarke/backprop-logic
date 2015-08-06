Backpropagation for Model Building
==================================

There's been a lot of interest recently in combining natural language
semantics with neural networks. One example of this is the excitement over memory
networks, as in (this paper)[http://www.aclweb.org/anthology/P/P15/P15-1150.pdf]
at ACL 2015 by Tai, Socher and Manning. 

Here I want to talk about something slightly different, although the
goal is similar, namely to reason about natural language
semantics. The idea is that we can use the same technique used to
train neural networks, backpropagation, to do something completely
different: model building. Model building solves the converse problem
to (automated theorem proving)[https://en.wikipedia.org/wiki/Automated_theorem_proving].
A model builder is useful when you want to know if a logical statement
is (satisfiable)[https://en.wikipedia.org/wiki/Satisfiability], but
they have also been used to solve problems in natural language
processing, such as (textual entailment)[http://www.let.rug.nl/bos/pubs/BosMarkert2006MLCW.pdf].

These two things don't immediately seem compatible, as backpropagation
relies on continuous vector space representations, whereas logic is
typically binary. The solution, of course, is to consider nonstandard
logics, such as fuzzy logics which can be interpreted in terms of real
numbers. The idea of combining fuzzy logic and neural networks
(is an old one)[https://en.wikipedia.org/wiki/Neuro-fuzzy], but we're
doing something slightly different here.
