# Sequence Diagram: Request Flow

Dựa vào `trace_id` `1a901331-cd8f-45ba-8c9c-70e1b769d657` được ghi nhận trong logs, luồng xử lý (request flow) của hệ thống A2A được mô tả như sau:

```mermaid
sequenceDiagram
    participant User as Test Client
    participant CA as Customer Agent
    participant Reg as Registry
    participant LA as Law Agent
    participant TA as Tax Agent
    participant CompA as Compliance Agent
    
    User->>CA: Send query
    Note over CA: Trace ID initiated:<br/>1a901331-cd8f-45ba-8c9c-70e1b769d657
    CA->>Reg: Discover "legal_question"
    Reg-->>CA: Return Law Agent address
    CA->>LA: Forward Query via A2A
    
    LA->>Reg: Discover "tax_question"
    Reg-->>LA: Return Tax Agent address
    LA->>TA: Send Tax sub-task
    
    LA->>Reg: Discover "compliance_question"
    Reg-->>LA: Return Compliance Agent address
    LA->>CompA: Send Compliance sub-task
    
    TA-->>LA: Tax Analysis
    CompA-->>LA: Compliance Analysis
    
    Note over LA: Aggregate all analyses
    LA-->>CA: Final Legal Report
    CA-->>User: Final Output
```

## Bài tập 5.1: Trace Request Flow
Các Agent trong hệ thống giao tiếp với nhau bằng giao thức HTTP phân tán. Dù khác Process và Port, nhưng chúng đều truyền chung một `trace_id` trong HTTP Header để đồng bộ hoá phiên thảo luận. Nhờ có `trace_id` này, chúng ta có thể dễ dàng xâu chuỗi Log của 5 server lại thành một luồng (flow) thống nhất.
