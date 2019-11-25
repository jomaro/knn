# an experiment of the K-Nearest Neighbors using Rust


You should have no hope that this will be useful to you.

But you are free to try if you wish.


## usage

```
$ cargo build --release
$ ./target/release/knn data/test1.knn 5 data/test1-k5.knn
```

## know improvements 

[ ] remove the call to `viz.sort_by` inside `predict`, use a simple round of insertion sort instead
[ ] improve naming
