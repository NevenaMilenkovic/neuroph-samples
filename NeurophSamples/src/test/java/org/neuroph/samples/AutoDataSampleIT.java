/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.neuroph.samples;

import java.util.List;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.nnet.Adaline;
import org.neuroph.nnet.learning.LMS;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Altos
 */
public class AutoDataSampleIT {

    DataSet trainingSet;
    DataSet testSet;

    public AutoDataSampleIT() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
        String dataSetFileName = "data_sets/autodata.txt";
        int inputsCount = 1;
        int outputsCount = 1;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(dataSetFileName, inputsCount, outputsCount, ",", false);
        Normalizer norm = new MaxNormalizer();
        norm.normalize(dataSet);
        dataSet.shuffle();

        List<DataSet> subSets = dataSet.split(60, 40);
        trainingSet = subSets.get(0);
        testSet = subSets.get(1);
    }

    @After
    public void tearDown() {
    }

    // Checks if the number of neural network iterations is smaller or equal to maximum number of iteration that is set.
    @Test
    public void testMaxIterations() {
        Adaline neuralNet = new Adaline(1);

        neuralNet.setLearningRule(new LMS());
        LMS learningRule = (LMS) neuralNet.getLearningRule();

        learningRule.setMaxIterations(1000);

        neuralNet.learn(trainingSet);

        assertTrue(learningRule.getCurrentIteration() <= learningRule.getMaxIterations());
    }

    // Checks if neural network error is smaller than maximum error that is set.
    @Test
    public void testMaxError() {
        Adaline neuralNet = new Adaline(1);

        neuralNet.setLearningRule(new LMS());
        LMS learningRule = (LMS) neuralNet.getLearningRule();

        learningRule.setMaxError(0.3);

        neuralNet.learn(trainingSet);

        assertTrue(learningRule.getTotalNetworkError() < learningRule.getMaxError());
    }

    // Checks if neural network stops training before it reaches maximum number of iterations that is set.
    // If number of iterations is smaller than maximim number of iteration it means that stop condition is maximum number of iteration.
    // If not, than stop condition is maximum network error.
    @Test
    public void testStopCondition() {
        Adaline neuralNet = new Adaline(1);

        neuralNet.setLearningRule(new LMS());
        LMS learningRule = (LMS) neuralNet.getLearningRule();

        learningRule.setMaxIterations(1000);

        neuralNet.learn(trainingSet);

        if (learningRule.getCurrentIteration() < learningRule.getMaxIterations()) {
            assertTrue(learningRule.getCurrentIteration() < learningRule.getMaxIterations());
            System.out.println("Stop condition is maximum error");
        } else {
            assertTrue(learningRule.getCurrentIteration() == learningRule.getMaxIterations());
            System.out.println("Stop condition is maximum number of iterations.");
        }
    }
}
