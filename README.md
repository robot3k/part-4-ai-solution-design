# Part 4: AI Solution Design for a Business Problem

## Overview

This part presents a complete AI solution design as an **AI Business Analyst**, addressing a real-world business problem using NLP-based text classification. The deliverable is a structured solution report — not a trained model, but a thorough design blueprint that covers problem definition, data requirements, model selection, evaluation planning, and responsible AI considerations.

## Selected Domain: Customer Support (Telecom)

**Business Problem:** Telecom customer support teams receive thousands of tickets daily with no automated sentiment-based prioritization. Angry, at-risk customers wait in the same queue as general inquiries, leading to slow response times and preventable churn.

**AI Solution:** Real-time sentiment classification of incoming support tickets (positive / neutral / negative) to enable automatic routing — negative-sentiment tickets go directly to the retention team, urgent ones are flagged for immediate response.

## AI Task Type: Text Classification

Each ticket message is classified into one of three sentiment categories. This is a **multi-class NLP classification** problem suited for transformer-based architectures.

## Reference Files Used

| File | Purpose |
|------|---------|
| `ai_usecase_reference_catalog.csv` | Identified "Customer Support / Ticket Sentiment Routing" use case |
| `business_kpi_sample.csv` | Baseline KPIs used to set AI performance targets |

**Selected catalog row:** Domain = "Customer Support", AI task = "Text Classification", Candidate model = "LSTM / Transformer model", Evaluation = "F1-score, resolution time"

## Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Model | Fine-tuned BERT | Best-in-class NLP accuracy; transfers rich pre-trained linguistic knowledge |
| Primary metric | Weighted F1-score | Handles any class imbalance in production data |
| Priority metric | Recall (negative class) | Missing an angry customer is costlier than a false positive |
| Human fallback | Confidence < 0.75 → human review | Prevents silent failures on edge cases |

## Expected Impact

| KPI | Before AI | After AI (Target) |
|-----|-----------|-------------------|
| Avg. first response time | 35 hours | < 15 hours |
| Customer satisfaction score | 6.8 / 10 | > 7.5 / 10 |
| Manual routing hours | ~500 hrs/month | < 100 hrs/month |
| Churn from negative tickets | Baseline | −15% |

## Repository Structure

```
part-4-ai-solution-design/
├── README.md
├── solution_report.md
└── diagrams/
    └── solution_architecture.png
```

## Files

- **`solution_report.md`** — Full 8-task solution design (2,500+ words)
- **`diagrams/solution_architecture.png`** — End-to-end system architecture diagram
