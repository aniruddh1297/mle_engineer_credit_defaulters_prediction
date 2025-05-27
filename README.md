# ML Azure Interview Task - 1 Week Take-Home Challenge

This take-home assignment is designed to evaluate your ability to build an end-to-end machine learning solution using Azure Machine Learning. You have **1 week** to complete this task and will present your work in a **30-minute presentation** during the interview.

## Timeline
- **Preparation Time**: 1 week from receipt of this task
- **Presentation**: 30 minutes during your interview session
- **Q&A Session**: Additional time following the presentation for technical discussion

## Objective

Build a complete machine learning pipeline using Azure Machine Learning that predicts whether a credit card client will default on their payment next month. Your solution should demonstrate production-ready ML practices, clear architectural decisions, and thoughtful approach to model deployment considerations.

## Project Structure

A starter folder structure is provided. You may modify or extend this structure as needed, ensuring the final solution is well-organized, documented, and reproducible.

## Dataset

Use the [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Link: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

**Target Variable**: "default payment next month"
- `1` = Default
- `0` = No default

## Core Requirements

### 1. Data Pipeline (`src/preprocess.py`)
- Load and explore the dataset
- Implement data cleaning and preprocessing
- Handle missing values, outliers, and data quality issues
- Apply feature engineering techniques
- Split data appropriately for training/validation/testing

### 2. Model Development (`src/train.py`)
- Implement and compare multiple classification algorithms
- Apply proper cross-validation and hyperparameter tuning
- Log experiments and metrics using Azure ML tracking
- Justify your model selection approach

### 3. Model Evaluation (`src/evaluate.py`)
- Comprehensive model evaluation with appropriate metrics
- Generate performance visualizations and interpretation
- Include business-relevant evaluation (e.g., cost-benefit analysis)
- Document model limitations and potential biases

### 4. Azure ML Pipeline (`pipeline/run_pipeline.py`)
- Create a complete Azure ML pipeline using SDK v2
- Orchestrate: Data Preprocessing → Training → Evaluation → Registration
- Implement proper error handling and logging
- Make the pipeline parameterizable and reusable

### 5. Infrastructure Setup
- Configure Azure ML workspace and compute resources
- Document your resource choices and cost considerations
- Ensure reproducibility across environments

## Advanced Topics for Discussion (Choose 2-3)

Select 2-3 advanced topics to **discuss and analyze** in your presentation (implementation not required):

- **Model Monitoring**: Data drift detection strategies and model performance monitoring approaches
- **Deployment Architecture**: Real-time vs batch inference design decisions and trade-offs
- **MLOps Integration**: Automated retraining triggers, CI/CD pipeline considerations
- **Model Explainability**: Interpretability requirements and implementation approaches (SHAP, LIME)
- **Advanced ML Techniques**: Feature stores, automated feature engineering, or ensemble methods
- **Security & Compliance**: Authentication, data encryption, and regulatory compliance considerations

## Presentation Requirements

Your 30-minute presentation should cover:

### Technical Implementation (20 minutes)
1. **Architecture Overview**
   - Solution architecture diagram
   - Azure services used and rationale

2. **Data & Feature Engineering**
   - Data exploration insights
   - Feature engineering decisions
   - Data quality considerations

3. **Model Development**
   - Model selection rationale
   - Training approach and results
   - Performance metrics and interpretation

4. **Pipeline & Infrastructure**
   - Azure ML pipeline design
   - Compute and cost optimization
   - Reproducibility measures

5. **Advanced Topics Discussion**
   - Analysis of your chosen advanced topics
   - Implementation approaches and trade-offs

### Business & Production Readiness (10 minutes)
6. **Production Strategy**
   - Deployment architecture recommendations
   - Monitoring and maintenance strategy
   - Risk assessment and mitigation

7. **Business Impact**
   - Model performance in business terms
   - Cost-benefit analysis
   - Recommendations for stakeholders

### Q&A Session (Following the presentation)
- Technical deep-dive into your implementation decisions
- Discussion of alternative approaches and trade-offs
- Code walkthrough if requested

## Deliverables

### 1. Code Package (Zip file containing):
- Complete source code with clear documentation
- Requirements/environment files
- README with setup instructions
- Configuration files for Azure ML

### 2. Technical Documentation
- Architecture decisions and rationale
- Model performance report
- Instructions for reproducing your results
- Cost analysis and optimization notes

### 3. Presentation Materials
- Slides for your 30-minute presentation
- Architecture diagrams
- Performance visualizations
- Screenshots of Azure ML workspace/pipelines

Please send the zip-file containing your work to HR for further evaluation.

## Evaluation Criteria

You will be assessed on:

**Technical Excellence**
- Code quality, structure, and documentation
- Proper use of Azure ML services
- Implementation of ML best practices
- Pipeline design and reproducibility

**Problem-Solving Approach**
- Data analysis and feature engineering quality
- Model selection and validation methodology
- Handling of edge cases and potential issues
- Innovation in approach

**Production Readiness**
- Scalability and maintainability considerations
- Monitoring and deployment strategy
- Cost optimization and resource management
- Security and compliance awareness

**Communication**
- Clarity of presentation and documentation
- Ability to explain technical concepts
- Business impact articulation
- Response to technical questions

## Setup and Support

### Azure Resources
- Please use a free azure subscription to perform the task

### Questions and Clarifications
- Technical questions can be sent via email
- Response time: Within 24 hours on business days
- No implementation guidance will be provided

## Tips for Success

1. **Start Early**: Set up your Azure environment on day 1
2. **Focus on Fundamentals**: Ensure core requirements work perfectly before adding advanced features
3. **Document Everything**: Clear documentation demonstrates professional development practices
4. **Practice Your Presentation**: Time yourself and prepare for technical questions
5. **Think Production-First**: Consider scalability, monitoring, and maintenance from the beginning
6. **Show Your Thinking**: Explain your decisions and trade-offs clearly

Good luck! We're excited to see your approach to this machine learning challenge.