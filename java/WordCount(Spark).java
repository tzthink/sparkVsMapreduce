package org.apache.spark.examples;

import scala.Tuple2;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

public final class JavaWordCount {
  private static final Pattern SPACE = Pattern.compile(" ");
  // define the skip patterns

  public static void main(String[] args) throws Exception {

    if (args.length < 1) {
      System.err.println("No import file");
      System.exit(1);
    }
    // if not input file, exit.

    SparkConf sparkConf = new SparkConf().setAppName("JavaWordCount");  
    JavaSparkContext ctx = new JavaSparkContext(sparkConf);
    // set up spark context

    JavaRDD<String> lines = ctx.textFile(args[0], 1);
    // transform input file to build RDD with lines  

    JavaRDD<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
      @Override
      public Iterable<String> call(String s) {
        return Arrays.asList(SPACE.split(s));
      }
    });
    // cut lines into words , also transform RDD

    JavaPairRDD<String, Integer> ones = words.mapToPair(new PairFunction<String, String, Integer>() {
      @Override
      public Tuple2<String, Integer> call(String s) {
        return new Tuple2<String, Integer>(s, 1);
      }
    });
    // map the <word, number of occrance> , also transform RDD

    JavaPairRDD<String, Integer> counts = ones.reduceByKey(new Function2<Integer, Integer, Integer>() {
      @Override
      public Integer call(Integer i1, Integer i2) {
        return i1 + i2;
      }
    });
    // count all the occurance transform RDD

    List<Tuple2<String, Integer>> output = counts.collect();
    ctx.stop();
    // collect and output the result.
  }
}