(ns text-mining-and-analytics.core
  (:require [clojure.pprint :refer [pprint]]))

(defn map-vals [f m]
  (into {} (for [[k v] m] [k (f v)])))

(defn log2 [n]
  (if (= 0 n) 0 (/ (Math/log n) (Math/log 2))))

(defn new-idf-model []
  {:document-count 0
   :document-frequencies {}})

(defn new-bm25-model []
  {:document-count 0
   :total-document-term-count 0})

(defn term-frequencies [document]
  (frequencies document))

(defn document-frequencies [document]
  (reduce (fn [document-frequencies term] (assoc document-frequencies term 1))
          {}
          document))

(defn document-term-count [document]
  (count document))

(defn document->idf-model [document]
  {:document-count 1
   :document-frequencies (document-frequencies document)})

(defn document->bm25-model [document]
  {:document-count 1
   :total-document-term-count (document-term-count document)})

(defn combine-idf-models [a b]
  {:document-count (apply + ((juxt a b) :document-count))
   :document-frequencies
   (apply merge-with + ((juxt a b) :document-frequencies))})

(defn combine-bm25-models [a b]
  {:document-count (apply + ((juxt a b) :document-count))
   :total-document-term-count
   (apply + ((juxt a b) :total-document-term-count))})

(defn make-model [combiner mapper documents]
  (reduce combiner (map mapper documents)))

(defn idf-model [documents]
  (make-model combine-idf-models document->idf-model documents))

(defn bm25-model [documents]
  (make-model combine-bm25-models document->bm25-model documents))

(def score-comparator (comparator (fn [x y] (> (second x) (second y)))))

(defn bm25-score
  [term-frequency document-term-count average-document-term-count k b]
  (let [[term frequency] term-frequency]
    [term (/ (* (+ k 1) frequency)
             (+ frequency (* k (+ (- 1 b)
                                  (* b (/ document-term-count
                                          average-document-term-count))))))]))

(defn bm25-scores [bm25-model term-frequencies]
  (let [k 1.25
        b 0.75
        document-term-count (count term-frequencies)
        average-document-term-count (average-document-term-count bm25-model)
        term-scores
        (map #(bm25-score % document-term-count average-document-term-count k b)
             term-frequencies)
        total-bm25-score (reduce + (map second term-scores))]
    (sort score-comparator
          (map (fn [[term score]]
                 [term (/ score total-bm25-score)])
               term-scores))))

(defn inverse-document-frequency [document-count document-frequency]
  (Math/log10 (/ (+ 1 document-count) document-frequency)))

(defn idf-score [idf-model term-frequency]
  (let [[term frequency] term-frequency
        df (get-in idf-model [:document-frequencies term] 1)]
    [term (inverse-document-frequency (:document-count idf-model) df)]))

(defn idf-scores [model term-frequencies]
  (sort score-comparator (map (partial idf-score model) term-frequencies)))

(defn tf-idf-score [idf-model term-frequency]
  (let [[term frequency] term-frequency]
    [term (* frequency (second (idf-score idf-model term-frequency)))]))

(defn tf-idf-scores [idf-model term-frequencies]
  (sort score-comparator
        (map (partial tf-idf-score idf-model) term-frequencies)))

(defn idf-weighted-bm25-scores [bm25-model idf-model term-frequencies]
  (let [bm25-scores* (into {} (bm25-scores bm25-model term-frequencies))
        idf-scores* (into {} (idf-scores idf-model term-frequencies))]
    (sort score-comparator (vec (merge-with * bm25-scores* idf-scores*)))))

(defn average-document-term-count [bm25-model]
  (/ (:total-document-term-count bm25-model) (:document-count bm25-model)))

(defn term-frequency [document term]
  (get (term-frequencies document) term))

(defn context [document term]
  (let [term-index (.indexOf document term)]
    ((juxt #(get % (- term-index 1)) #(get % (+ term-index 1))) document)))

(defn term-probability [document term]
  (/ (get (term-frequencies document) term 0) (count document)))

(defn term-probability-given [document term given]
  (let [given-term-probability (term-probability document given)]
    (if (= 0 given-term-probability)
      0
      (/ (* (term-probability document term)
            given-term-probability)
         given-term-probability))))

(defn normalize-term-frequencies [a b]
  (let [normalize-term-frequencies* (fn [a b]
                                      (merge (map-vals (constantly 0) b) a))]
    (->> [a b]
         (map term-frequencies)
         (apply normalize-term-frequencies*))))

(defn document-similarity [bm25-model idf-model a b]
  (->> [(normalize-term-frequencies a b) (normalize-term-frequencies b a)]
       (map (partial idf-weighted-bm25-scores bm25-model idf-model))
       (map (partial into {}))
       (apply merge-with *)
       (vals)
       (reduce +)))

(defn term-probabilities [document term]
  (let [term-present-probability (term-probability document term)
        term-absent-probability (- 1 term-present-probability)]
    [term-absent-probability term-present-probability]))

(defn term-probabilities-given [document term given]
  (let [[term-0 term-1] (term-probabilities document term)
        [given-0 given-1] (term-probabilities document given)
        term-1-given-1 (term-probability-given document term given)
        term-1-given-0 (- term-1 term-1-given-1)
        term-0-given-1 (- given-1 term-1-given-1)
        term-0-given-0 (- 1 term-1-given-1 term-1-given-0 term-0-given-1)]
    [term-0-given-0 term-0-given-1 term-1-given-0 term-1-given-1]))

(defn entropy [document term]
  (->> (term-probabilities document term)
       (map #(if (= 0 %) 0 (* % (log2 (/ 1 %)))))
       (reduce +)))

(defn conditional-entropy [document term given]
  (let [[given-0 given-1] (term-probabilities document given)
        [term-0-given-0 term-0-given-1 term-1-given-0 term-1-given-1]
        (term-probabilities-given document term given)]
    (+ (* given-0
          (+ (* -1 term-0-given-0 (log2 term-0-given-0))
             (* -1 term-1-given-0 (log2 term-1-given-0))))
       (* given-1
          (+ (* -1 term-0-given-1 (log2 term-0-given-1))
             (* -1 term-1-given-1 (log2 term-1-given-1)))))))

(defn mutual-information [document term-a term-b]
  (- (entropy document term-a) (conditional-entropy document term-a term-b)))

(def corpus
  [["my" "cat" "eats" "fish" "on" "Saturday" "after" "the" "news" "is" "on"]
   ["my" "cat" "eats" "turkey" "on" "Saturday"]
   ["my" "dog" "eats" "meat" "on" "Saturday" "nights"]
   ["my" "dog" "eats" "turkey" "on" "dog" "Saturday"]
   ["why" "do" "you" "think" "this" "is" "on" "you"]
   ["why" "do" "play" "you" "you" "you" "would" "ever" "on" "you" "you" "you"]])

(def bm25-model* (bm25-model corpus))
(def idf-model* (idf-model corpus))

(comment
 (pprint
  (document-similarity bm25-model* idf-model* (get corpus 2) (get corpus 1)))

 (pprint
  (term-probabilities (get corpus 5) "you"))

 (pprint
  (term-probabilities-given (get corpus 5) "do" "for"))

 (pprint
  (entropy (get corpus 5) "you"))

 (pprint
  (conditional-entropy (get corpus 5) "you" "on"))

 (pprint
  (mutual-information (get corpus 5) "on" "why"))

 (pprint
  (->> corpus
       (map term-frequencies)
       (map (partial idf-weighted-bm25-scores bm25-model* idf-model*)))))
