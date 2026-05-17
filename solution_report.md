# AI Solution Design Report
## Customer Support Ticket Sentiment Routing — Telecom Domain

**Prepared by:** Module 5 Assignment, Part 4  
**Domain Selected:** Customer Support (Telecom)  
**AI Task Type:** Text Classification (Sentiment Analysis)

---

## Task 1: Business Domain

**Selected Domain:** Customer Support — Telecom Industry

Telecom companies handle tens of thousands of customer support interactions daily across chat, phone, email, and social media. The sheer volume makes it impossible for human agents to manually read and prioritize every ticket before routing it to the appropriate team.

---

## Task 2: Business Problem Definition

### Problem Being Solved

Customer support tickets arrive with no structured metadata about their emotional urgency. An angry customer threatening to cancel (negative sentiment, urgent) gets queued identically to a curious customer asking about plan upgrades (neutral, low urgency). This results in:
- High-value at-risk customers waiting too long and churning
- Urgent complaints being resolved too slowly, escalating to social media complaints
- Agent time wasted on low-urgency tickets when critical ones are pending

### Stakeholders
- **Tier-1 Support Agents:** Need pre-routed tickets so they can focus immediately on urgent cases
- **Customer Retention Team:** Need early warning on at-risk negative sentiment customers
- **Operations Manager:** Needs KPIs on resolution time and satisfaction trends
- **End Customers:** Need fast, appropriate responses to their specific concern

### Current Manual Process
1. Agent opens support queue — all tickets shown chronologically or by arbitrary priority
2. Agent reads through tickets to gauge urgency (mentally classifying sentiment)
3. Routes ticket manually to appropriate sub-queue (billing, technical, cancellations)
4. Average time from ticket creation to first response: **35 hours** (from KPI data)

### Limitations of Current Process
- Inconsistent classification — different agents assign different priority levels to the same message
- No automatic escalation for high-emotion negative tickets
- Batch processing means a critical ticket at 11:55 PM waits until the next morning
- No data trail for analyzing sentiment trends over time

---

## Task 3: AI Task Type — Text Classification

**Chosen Type:** Multi-class Text Classification (Sentiment Analysis)

**Why this fits:**
- Each incoming customer message must be assigned exactly one sentiment label: `positive`, `neutral`, or `negative`
- The labels directly map to routing rules (negative → retention team, urgent flag → priority queue)
- The input is unstructured text — standard tabular ML doesn't apply
- There are clear class boundaries that a trained model can learn

**Why not other types?**
- *Sequence prediction:* We don't need to predict the next word; we need a label for the whole message
- *Regression:* Sentiment is categorical, not a continuous scale in this use case
- *Object detection:* Not applicable — no image data involved

---

## Task 4: Data Requirement Plan

### Type of Data
**Unstructured Text** (primary) + **Structured Metadata** (secondary)

### Input Features
| Feature | Type | Description |
|---------|------|-------------|
| `customer_message` | Text | Raw message content |
| `channel` | Categorical | chat / phone / email / social |
| `word_count` | Numerical | Derived message length |
| `urgent_flag` | Binary | System-flagged urgency keywords |
| `time_of_day` | Categorical | Morning / afternoon / evening / night |
| `customer_tenure_months` | Numerical | From CRM integration |
| `recent_ticket_count` | Numerical | Tickets in last 30 days |

### Target Variable
- `sentiment_label`: positive / neutral / negative (3-class categorical)

### Data Collection Method
1. **Historical tickets:** Export 3–5 years of resolved tickets from CRM (ServiceNow, Zendesk, etc.)
2. **Human labeling:** Have 2–3 agents independently label a sample of 5,000 tickets; use majority vote for ground truth
3. **Active learning:** As the model is deployed, periodically label borderline predictions to improve training data

### Data Volume Requirement
- Minimum: 10,000 labeled tickets per class for robust training
- Recommended: 50,000+ for transformer fine-tuning

### Data Quality Risks
| Risk | Mitigation |
|------|-----------|
| Label noise from subjective sentiment interpretation | Inter-annotator agreement check (Cohen's Kappa > 0.7) |
| Language drift (new slang, product terms) | Monthly model retraining with new tickets |
| Channel bias (phone transcripts vs. written chat) | Stratified sampling across channels |
| Class imbalance (neutral often overrepresented) | Oversampling / class-weighted loss during training |

---

## Task 5: Model Recommendation

### Recommended Architecture: Fine-tuned BERT (Bidirectional Encoder Representations from Transformers)

**Justification:**
BERT pre-trains on billions of documents and learns deep contextual word representations. When fine-tuned on our labeled support tickets (10,000+ examples), it achieves state-of-the-art text classification with relatively little domain-specific training data.

```
Customer Message Text
    ↓
BERT Tokenizer (WordPiece, max_length=128)
    ↓
BERT Encoder (12 layers of bidirectional self-attention)
    ↓
[CLS] token representation → Dense(128, ReLU) → Dropout(0.1)
    ↓
Dense(3, Softmax) → [negative, neutral, positive]
```

**Alternative models considered:**
| Model | Pro | Con |
|-------|-----|-----|
| Logistic Regression + TF-IDF | Fast, interpretable | Ignores word order, misses context |
| LSTM | Captures sequence | Slow to train, lower ceiling than transformers |
| **BERT (recommended)** | Best accuracy, bidirectional context | Requires GPU for fine-tuning, larger model size |
| DistilBERT | 40% smaller, 60% faster than BERT | Slight accuracy drop (~3%) |

**For production:** If latency is critical (< 50ms response), use **DistilBERT**. If accuracy is paramount (routing high-value customers), use full **BERT-base**.

---

## Task 6: Evaluation Plan

### Technical Metrics
| Metric | Target | Rationale |
|--------|--------|-----------|
| Weighted F1-score | > 0.90 | Handles any residual class imbalance |
| Recall (negative class) | > 0.92 | Missing an angry customer is more costly than a false positive |
| Precision (overall) | > 0.88 | Minimize incorrect routing |
| Inference latency | < 100ms | Acceptable for real-time ticket routing |

### Business Metrics
| Metric | Baseline (Manual) | AI Target |
|--------|------------------|-----------|
| Average first response time | 35 hours | < 15 hours |
| Customer satisfaction score (CSAT) | 6.8 / 10 | > 7.5 / 10 |
| Churn rate (negative-sentiment customers) | Baseline TBD | -15% reduction |
| Agent hours on manual routing | ~500 hrs/month | < 100 hrs/month |

### Possible Failure Cases
- **Sarcasm:** "Oh great, another billing error" — may be classified as positive
- **Mixed sentiment:** "Love the product, but the support is terrible" — ambiguous
- **Code-switching:** Hindi-English mixed messages common in Indian telecom may not tokenize well for an English BERT
- **Novel complaint types:** A new issue (e.g., new app bug) may not match training distribution

### Human Review and Validation Process
1. **Confidence threshold:** If model confidence < 0.75, route to human review queue
2. **Weekly audit:** Sample 200 auto-classified tickets; agent validates correctness
3. **Feedback loop:** Agent corrections feed back into retraining dataset monthly
4. **Escalation override:** Agents can always manually re-label tickets; these corrections are logged

---

## Task 7: Responsible AI Considerations

### Bias Risks
- **Language bias:** If training data is predominantly English, the model will perform poorly on regional language inputs. Indian telecom customers often write in Hinglish — the model must be tested on multilingual inputs.
- **Channel bias:** Customers who call (voice-to-text transcription) may have different writing patterns than chat customers. If phone transcripts are under-represented in training data, the model may underperform on that channel.
- **Historical labeling bias:** If human agents historically marked certain customer groups as "negative" more frequently due to stereotyping, the model will learn and amplify that bias.

### Risk of Incorrect Predictions
- A frustrated customer classified as "neutral" misses the retention queue and churns — direct revenue loss
- A neutral query flagged as "negative" wastes retention agent time
- Mitigation: Use confidence scores + human fallback for borderline predictions

### Privacy Concerns
- Customer messages may contain PII (name, address, account number). The model should not store raw message text in logs beyond what compliance requires.
- Ensure GDPR / DPDP (India) compliance: customers must be informed their messages are processed by AI
- PII should be masked before the message is passed to the model API

### Over-Reliance on AI
- Teams may stop reading actual tickets, missing context the model can't capture (tone of voice in transcripts, customer history context)
- Mitigation: AI routes and prioritizes, but humans make final decisions and responses

### Impact on Workers
- Concern: Automating routing may reduce headcount for manual sorters
- Mitigation: Re-deploy agents to higher-value tasks (complex case resolution, retention calls) where human empathy is irreplaceable

### Need for Human Oversight
- Monthly model retraining review by a data scientist
- Quarterly bias audit across channels, regions, and customer demographics
- A "model card" should be maintained documenting training data, known limitations, and performance metrics

---

## Task 8: Final Solution Summary

| Component | Detail |
|-----------|--------|
| **Problem** | Manual support ticket routing causes slow response times and inconsistent prioritization for angry/at-risk customers |
| **Proposed AI Solution** | Real-time sentiment classification (positive / neutral / negative) of incoming support tickets, with automatic routing to the appropriate queue |
| **Required Data** | 10,000+ labeled historical tickets per class; metadata (channel, tenure, ticket frequency) |
| **Model Recommendation** | Fine-tuned BERT-base (or DistilBERT for latency-sensitive deployment) |
| **Expected Business Impact** | First response time cut from 35 → 15 hours; CSAT from 6.8 → 7.5+; ~15% reduction in churn from high-risk negative-sentiment customers; 80% reduction in manual routing hours |
| **Risks** | Sarcasm misclassification, multilingual input gaps, PII in messages, language/channel bias in training data |
| **Mitigation Plan** | Confidence thresholds for human fallback; monthly retraining; multilingual testing; PII masking; weekly random audit |

> **Personal Reflection:** Designing this solution end-to-end — from problem statement to responsible AI risks — revealed how much more there is to deploying AI than just training a model. The hardest parts aren't technical: they're deciding what metrics actually matter to the business, anticipating how the system can fail in production, and building the human oversight processes that make AI trustworthy. A 99% F1 on a clean test set means nothing if the model fails silently on Hindi-English code-switched messages that make up 30% of real production traffic.
