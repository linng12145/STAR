# STAR

## Data Set

Data download address:
https://hub.marinecadastre.gov/pages/vesseltraffic

## Data PreProcess

```python
cd DataPreProcess
python gather.py
python test_process.py
```

## train

```python
cd detection_stage
python train.py
```

```python
cd recovery_stage
python train.py
```

## test

```python
cd recovery_stage
python test_TERI.py
```

## demo

```python
cd demo
python app.py
```
