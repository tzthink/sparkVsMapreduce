package org.apache.mahout.clustering.syntheticcontrol.kmeans;

import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.canopy.CanopyDriver;
import org.apache.mahout.clustering.conversion.InputDriver;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.RandomSeedGenerator;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.utils.clustering.ClusterDumper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job extends AbstractJob {
  
  private static final Logger log = LoggerFactory.getLogger(Job.class);
  
  private static final String DIRECTORY_CONTAINING_CONVERTED_INPUT = "data";
  
  private Job() {
  }
  
  public static void main(String[] args) throws Exception {
    if (args.length > 0) {
      log.info("Running with only user-supplied arguments");
      ToolRunner.run(new Configuration(), new Job(), args);
    } else {
      log.info("Running with default arguments");
      Path output = new Path("output");
      Configuration conf = new Configuration();
      HadoopUtil.delete(conf, output);
      run(conf, new Path("testdata"), output, new EuclideanDistanceMeasure(), 6, 0.5, 10);
    }
  }
  // main functions
  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.numClustersOption().create());
    addOption(DefaultOptionCreator.t1Option().create());
    addOption(DefaultOptionCreator.t2Option().create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    // get the option argument
    Map<String,List<String>> argMap = parseArguments(args);
    if (argMap == null) {
      return -1;
    }
    // parse argument
    Path input = getInputPath();
    Path output = getOutputPath();
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = SquaredEuclideanDistanceMeasure.class.getName();
    }
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);
    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      int k = Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION));
      run(getConf(), input, output, measure, k, convergenceDelta, maxIterations);
    } else {
      double t1 = Double.parseDouble(getOption(DefaultOptionCreator.T1_OPTION));
      double t2 = Double.parseDouble(getOption(DefaultOptionCreator.T2_OPTION));
      run(getConf(), input, output, measure, t1, t2, convergenceDelta, maxIterations);
    }
    return 0;
  }
  // deal with the option argument
  public static void run(Configuration conf, Path input, Path output, DistanceMeasure measure, int k,
      double convergenceDelta, int maxIterations) throws Exception {
    Path directoryContainingConvertedInput = new Path(output, DIRECTORY_CONTAINING_CONVERTED_INPUT);
    log.info("Preparing Input");
    InputDriver.runJob(input, directoryContainingConvertedInput, "org.apache.mahout.math.RandomAccessSparseVector");
    log.info("Running random seed to get initial clusters");
    Path clusters = new Path(output, "random-seeds");
    clusters = RandomSeedGenerator.buildRandom(conf, directoryContainingConvertedInput, clusters, k, measure);
    log.info("Running KMeans with k = {}", k);
    KMeansDriver.run(conf, directoryContainingConvertedInput, clusters, output, convergenceDelta,
        maxIterations, true, 0.0, false);
    // run ClusterDumper
    Path outGlob = new Path(output, "clusters-*-final");
    Path clusteredPoints = new Path(output,"clusteredPoints");
    log.info("Dumping out clusters from clusters: {} and clusteredPoints: {}", outGlob, clusteredPoints);
    ClusterDumper clusterDumper = new ClusterDumper(outGlob, clusteredPoints);
    clusterDumper.printClusters(null);
  }
  // running k -means with argument set 1
  public static void run(Configuration conf, Path input, Path output, DistanceMeasure measure, double t1, double t2,
      double convergenceDelta, int maxIterations) throws Exception {
    Path directoryContainingConvertedInput = new Path(output, DIRECTORY_CONTAINING_CONVERTED_INPUT);
    log.info("Preparing Input");
    InputDriver.runJob(input, directoryContainingConvertedInput, "org.apache.mahout.math.RandomAccessSparseVector");
    log.info("Running Canopy to get initial clusters");
    Path canopyOutput = new Path(output, "canopies");
    CanopyDriver.run(new Configuration(), directoryContainingConvertedInput, canopyOutput, measure, t1, t2, false, 0.0,
        false);
    log.info("Running KMeans");
    KMeansDriver.run(conf, directoryContainingConvertedInput, new Path(canopyOutput, Cluster.INITIAL_CLUSTERS_DIR
        + "-final"), output, convergenceDelta, maxIterations, true, 0.0, false);
    // run ClusterDumper
    ClusterDumper clusterDumper = new ClusterDumper(new Path(output, "clusters-*-final"), new Path(output,
        "clusteredPoints"));
    clusterDumper.printClusters(null);
  }
}  // running k -means with argument set 2
