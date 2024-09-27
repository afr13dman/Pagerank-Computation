# Pagerank Project

In this project, I created a simple search engine for the website <https://www.lawfareblog.com>, which provides legal analysis on US national security issues.

In order to create a simple search engine, we use PageRank to return on the most important results. The paper, *Deeper Inside Pagerank* discusses how PageRank begins with a matrix $P$ which is a square nxn matrix whose element $p_{ij}$ is the probability of moving from state i (page i) to state j (page j). Then, the matrix undergoes transformation to become a stochastic, irreducible, and primitive matrix $$\bar{\bar P}$$.

## Background

**Data:**

The `data` folder contains two files that store example "web graphs".
The file `small.csv.gz` contains the example graph from the *Deeper Inside Pagerank* paper. Since, this is a small graph, we can use the command `zcat` to decompress the gzipped file and then output the contents to manually inspect the graph.
```
$ zcat data/small.csv.gz
source,target
1,2
1,3
3,1
3,2
3,5
4,5
4,6
5,6
5,4
6,4
```

This graph is stored as a CSV file, where the first line is a header and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog for which we want to create the simple search engine. Here are the first 10 lines of the file:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```

**Code:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases. 

## Task 1: The Power Method

Within the function `WebGraph.power_method`, we implemented the power method for calculated the PageRank of each node in the graph. The paper *Deeper Inside Pagerank* provided us with the following definition for $x^k$, the Pagerank vector:
$` x^k = x^{k-1}\bar{\bar P} = \alpha x^{k-1}P + (\alpha x^{k-1} a + (1-\alpha ) v `$.
Here $a$ is a vector of 1s and 0s, where the ith entry is 1 if the corresponding node is a dangling node (a node with no outlinks), otherwise the entry is 0, and $v$ is a personalization vector which is a stochastic vector chosen by the user.

In the `WebGraph.power_method` function, we implented the equation $`x^k = \alpha x^{k-1}P + (\alpha x^{k-1} a + (1-\alpha ) v`$ because using the matrix $P$ allows us to do less computations and if $P$ is sparse, which it is, then the runtime will be much faster.

After implementing the `WebGraph.power_method` function, I examined the outcome for the following commands. We will see below that the file raises a `UserWarning` about a deprecated command. While this is important in the long term as the command will soon not be supported, for this project we will keep the same command, but the command `torch.sparse.SparseTensor()` can be replaced with `torch.sparse_coo_tensor()`.

**Task 1, part 1:**
```
$ python pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
DEBUG:root:i=0 residual=0.6277832984924316
DEBUG:root:i=1 residual=0.11841227114200592
DEBUG:root:i=2 residual=0.07070135325193405
DEBUG:root:i=3 residual=0.03181539848446846
DEBUG:root:i=4 residual=0.020496614277362823
DEBUG:root:i=5 residual=0.010108335874974728
DEBUG:root:i=6 residual=0.0063715302385389805
DEBUG:root:i=7 residual=0.0034228134900331497
DEBUG:root:i=8 residual=0.0020879723597317934
DEBUG:root:i=9 residual=0.0011749140685424209
DEBUG:root:i=10 residual=0.0007013367721810937
DEBUG:root:i=11 residual=0.0004032354918308556
DEBUG:root:i=12 residual=0.00023798183246981353
DEBUG:root:i=13 residual=0.00013813978875987232
DEBUG:root:i=14 residual=8.109924965538085e-05
DEBUG:root:i=15 residual=4.72828141937498e-05
DEBUG:root:i=16 residual=2.7671592761180364e-05
DEBUG:root:i=17 residual=1.614489156054333e-05
DEBUG:root:i=18 residual=9.430469617655035e-06
DEBUG:root:i=19 residual=5.500062798091676e-06
DEBUG:root:i=20 residual=3.2380467018811032e-06
DEBUG:root:i=21 residual=1.924154730659211e-06
DEBUG:root:i=22 residual=1.1138057516291155e-06
DEBUG:root:i=23 residual=6.438611990233767e-07
INFO:root:rank=0 pagerank=6.6270e-01 url=4
INFO:root:rank=1 pagerank=5.2179e-01 url=6
INFO:root:rank=2 pagerank=4.1434e-01 url=5
INFO:root:rank=3 pagerank=2.3175e-01 url=2
INFO:root:rank=4 pagerank=1.8590e-01 url=3
INFO:root:rank=5 pagerank=1.6917e-01 url=1
```


**Task 1, part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter and then will return all nodes that match the query string sorted according to their pagerank.

```
$ python pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9231e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0396e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9159e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7047e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6262e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5051e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3625e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1254e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0193e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
```
```
$ python pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
INFO:root:rank=0 pagerank=5.7828e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2341e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1299e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6601e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5936e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3073e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0937e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7593e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4510e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4486e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors
```
```
$ python pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
INFO:root:rank=0 pagerank=4.5748e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4176e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6929e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9393e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5453e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5358e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5259e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4222e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1464e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```

**Task 1, part 3:**

Since this graph includes many articles, how do we know if a link is a non-article page? There is no way to answer this with perfect accuracy but one way is to compute the "in-link ratio" of each node, which is the total number of edges with the node as a target divided by the total number of nodes. Then, we will remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

```
$ python pagerank.py --data=data/lawfareblog.csv.gz
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
```
```
$ python pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```

**Task 1, part 4:**

The runtime of pagerank depends heavily on the eigengap of the $\bar{\bar P}$ matrix, which is bounded by the alpha parameter. The graph $P$ for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence. A large alpha value implies that the structure of the webgraph has more influence on the final result, while a smaller alpha value ignores the structure of the webgraph.

Changing the value of alpha gives us very different pagerank rankings.

The following commands examine the affects of using the `--alpha` and `--filter_ratio` parameters:

```
$ python pagerank.py --data=data/lawfareblog.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
DEBUG:root:i=1 residual=0.11642322689294815
DEBUG:root:i=2 residual=0.0749533399939537
DEBUG:root:i=3 residual=0.03170184791088104
DEBUG:root:i=4 residual=0.01745055429637432
DEBUG:root:i=5 residual=0.008532325737178326
DEBUG:root:i=6 residual=0.00444810139015317
DEBUG:root:i=7 residual=0.002248029923066497
DEBUG:root:i=8 residual=0.0011543907457962632
DEBUG:root:i=9 residual=0.0005845933337695897
DEBUG:root:i=10 residual=0.0002967794134747237
...
DEBUG:root:i=995 residual=4.0947388697532006e-06
DEBUG:root:i=996 residual=4.078880920133088e-06
DEBUG:root:i=997 residual=4.072263436682988e-06
DEBUG:root:i=998 residual=4.0882628127292264e-06
DEBUG:root:i=999 residual=4.0947388697532006e-06
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
```
```
$ python pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
DEBUG:root:computing indices
DEBUG:root:computing values
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
DEBUG:root:i=0 residual=141.91505432128906
DEBUG:root:i=1 residual=0.0708821639418602
DEBUG:root:i=2 residual=0.01882285252213478
DEBUG:root:i=3 residual=0.006958338897675276
DEBUG:root:i=4 residual=0.002735838061198592
DEBUG:root:i=5 residual=0.0010345635237172246
DEBUG:root:i=6 residual=0.00037746329326182604
DEBUG:root:i=7 residual=0.0001353326952084899
DEBUG:root:i=8 residual=4.822362097911537e-05
DEBUG:root:i=9 residual=1.717347913654521e-05
DEBUG:root:i=10 residual=6.114406915003201e-06
DEBUG:root:i=11 residual=2.176922862417996e-06
DEBUG:root:i=12 residual=7.833889412722783e-07
INFO:root:rank=0 pagerank=2.8859e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=1 pagerank=2.8859e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=2 pagerank=2.8859e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=3 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-resources-related-travel-ban
INFO:root:rank=4 pagerank=2.8859e-01 url=www.lawfareblog.com/subscribe-lawfare
INFO:root:rank=5 pagerank=2.8859e-01 url=www.lawfareblog.com/topics
INFO:root:rank=6 pagerank=2.8859e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=7 pagerank=2.8859e-01 url=www.lawfareblog.com/our-comments-policy
INFO:root:rank=8 pagerank=2.8859e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8859e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
```
```
$ python pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
DEBUG:root:computing indices
DEBUG:root:computing values
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
DEBUG:root:i=0 residual=133.37046813964844
DEBUG:root:i=1 residual=0.4985728859901428
DEBUG:root:i=2 residual=0.134186252951622
DEBUG:root:i=3 residual=0.06922126561403275
DEBUG:root:i=4 residual=0.023409493267536163
DEBUG:root:i=5 residual=0.010187417268753052
DEBUG:root:i=6 residual=0.004906961694359779
DEBUG:root:i=7 residual=0.0022800215519964695
DEBUG:root:i=8 residual=0.001074681873433292
DEBUG:root:i=9 residual=0.00052502506878227
DEBUG:root:i=10 residual=0.00026971090119332075
DEBUG:root:i=11 residual=0.00014566551544703543
DEBUG:root:i=12 residual=8.228283695643768e-05
DEBUG:root:i=13 residual=4.811624239664525e-05
DEBUG:root:i=14 residual=2.8800663130823523e-05
DEBUG:root:i=15 residual=1.739492108754348e-05
DEBUG:root:i=16 residual=1.0555354492680635e-05
DEBUG:root:i=17 residual=6.3742777456354816e-06
DEBUG:root:i=18 residual=3.839234068436781e-06
DEBUG:root:i=19 residual=2.2957522105571115e-06
DEBUG:root:i=20 residual=1.3715186923946021e-06
DEBUG:root:i=21 residual=8.186000286514172e-07
INFO:root:rank=0 pagerank=3.4697e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9522e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=5 pagerank=1.5100e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=6 pagerank=1.5072e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4958e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
```
$ python pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
DEBUG:root:computing indices
DEBUG:root:computing values
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
DEBUG:root:i=0 residual=133.93057250976562
DEBUG:root:i=1 residual=0.5695652365684509
DEBUG:root:i=2 residual=0.38299503922462463
DEBUG:root:i=3 residual=0.21739308536052704
DEBUG:root:i=4 residual=0.1404506415128708
DEBUG:root:i=5 residual=0.10851356387138367
DEBUG:root:i=6 residual=0.09284130483865738
DEBUG:root:i=7 residual=0.08225560188293457
DEBUG:root:i=8 residual=0.07338891923427582
...
DEBUG:root:i=679 residual=1.0626848734318628e-06
DEBUG:root:i=680 residual=1.0508224477234762e-06
DEBUG:root:i=681 residual=1.0391307796453475e-06
DEBUG:root:i=682 residual=1.024263838189654e-06
DEBUG:root:i=683 residual=1.0239343737339368e-06
DEBUG:root:i=684 residual=1.0096704272655188e-06
DEBUG:root:i=685 residual=1.0010728601628216e-06
DEBUG:root:i=686 residual=9.88530473478022e-07
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0551e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1757e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2045e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6028e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6024e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6021e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6021e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

## Task 2: The Personalization Vector

The most interesting applications of pagerank involve the personalization vector. We implemented the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _index_to_url function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Task 2, part 1:**
```
$ python pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

**Task 2, part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
/Users/annef/Data_Mining/Pagerank/pagerank.py:73: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=). (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:643.)
  self.P = torch.sparse.FloatTensor(i, v, torch.Size([n,n]))
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-committee-holds-hearing-priorities-missile-defense
```

This commands outputs many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine", but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.
The vast majority of industry data mining work is finding clever uses of standard algorithms.
