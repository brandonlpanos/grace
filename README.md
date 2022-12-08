# grace

Discover possible lost geodesy gravitational signals using a Mutual Information Neural Estimator on residuals. The theory for MINE-networks can be found here: https://arxiv.org/pdf/1801.04062.pdf. For a description of Mutual Information see the book: Elements of Information Theory by Thomas Cover. PMI is bounded by -infinity and min( -log p(x), -log p(y) ). A PMI of =<0, means no dependancy, while a PMI >0 means a dependency (possible higher order correlation). We calculate PMI between location vectors and their associated scalar residuals.