# Securing the Rupee & the Euro: A 12-Month Playbook for India-EU Financial Cyber Resilience

## Executive Summary

### Regulatory Shockwave Incoming: DORA's January 2025 Deadline Creates Urgent Compliance Gaps
The EU's Digital Operational Resilience Act (DORA) becomes fully applicable on **January 17, 2025**, imposing a stringent, multi-stage incident reporting timeline (initial report within 4 hours of classification and no later than 24 hours from awareness, intermediate within 72 hours, and final within one month) on 20 types of financial entities [1] [2] [3]. This creates an immediate compliance conflict for Indian financial institutions servicing EU clients, who must also adhere to CERT-In's mandatory **6-hour** incident reporting window [4] [5]. Without a harmonized reporting protocol, these firms face the risk of dual-regime fines and potential data-freeze orders.

### Threat Volume Diverges, but Attacker Sophistication Converges
While threat volumes differ—the ECB noted a **54%** year-over-year increase in reported cyber incidents in 2020, and phishing attacks on India's BFSI sector surged by **175%** in H1 2024—the impact is converging [CITE_NOT_FOUND][CITE_NOT_FOUND]. Both blocs face sophisticated adversaries, including state-sponsored groups and ransomware-as-a-service (RaaS) operators like LockBit, RansomHub, and KillSec, who are increasingly using AI-powered tools and complex extortion tactics [6] [CITE_NOT_FOUND]. This parity in attacker capability underscores the need for joint threat-hunting exercises to reduce adversary dwell times.

### The Supply-Chain Blind Spot: Third-Party Breaches Prove Costlier and Harder to Contain
Recent supply-chain attacks demonstrate a critical, shared vulnerability. The **January 2023** ransomware attack on ION Group, a UK-based software provider, disrupted derivatives trading for major banks in the US and Europe, forcing manual trade processing and delaying regulatory reporting [7] [8] [9]. Similarly, the **June 2023** MOVEit vulnerability exposed data from over 1,000 organizations, including financial institutions and pension schemes that relied on payroll provider Zellis [10] [CITE_NOT_FOUND]. These incidents, which cost victims **11.8%** more and take **12.8%** longer to contain, highlight the urgent need for joint SBOM (Software Bill of Materials) attestations in vendor contracts and pooled audits for cloud-critical fintechs [10].

### The Widening Skills Gap Is Now a Systemic Risk
The EU faces an estimated shortage of **299,000** cybersecurity professionals, a **9%** increase from 2023, while India has less than half of the one million cybersecurity professionals it needs [11] [12]. This talent deficit poses a systemic risk, as it will bottleneck the implementation of mandatory advanced testing regimes like DORA's Threat-Led Penetration Testing (TLPT) and CERT-In's annual audits [13] [14]. A joint "Cyber Fellowship 500" program, focused on critical roles like threat hunting and red teaming, is a pragmatic step to build a shared talent pipeline.

### Fragmented Data-Sharing Rules Stall Real-Time Threat Intelligence
Conflicting data protection and incident reporting rules are a major barrier to effective cross-border threat intelligence sharing. The EU's GDPR mandates a 72-hour data breach notification window, while India's DPDP Act requires notification "without delay" in addition to CERT-In's 6-hour cyber incident reporting rule [CITE_NOT_FOUND][4] [15]. A unified TLP:RED lane, anchored in a formal MoU between EU-SCICF and CERT-In, can create a trusted channel to share Indicators of Compromise (IoCs) in minutes, not days, satisfying both privacy and security imperatives.

### Cloud Concentration Creates a Single Point of Failure
The hyper-concentration of cloud services, with AWS, Azure, and Google Cloud holding significant market share in both the EU and India, creates a critical single point of failure [CITE_NOT_FOUND]. DORA's CTPP oversight framework and the RBI's mandate for tested exit strategies with stringent RTO/RPO targets (as low as 15-minute RTO and near-zero RPO for some systems) reflect a shared regulatory concern [3] [16]. Jointly piloting portable reference architectures using Kubernetes and conducting annual exit drills are essential to mitigate this systemic risk.

### Zero Trust and AI-Assisted SOCs Are Deployment-Ready and Delivering ROI
Zero Trust Architecture (ZTA) is now a regulatory expectation in India, and both Indian and EU banks are actively planning to roll out AI-driven XDR and SOAR platforms in 2025 [17] [CITE_NOT_FOUND]. Financial institutions already leveraging AI in their security protocols report significantly lower data breach costs (average **US$1.76 million** less) and faster detection and containment times (**108 days** quicker) [18]. A joint sandbox for AI-guided incident command, governed by the EU AI Act's "high-risk" controls and RBI's "AI-aware defense" mandate, can validate ROI and establish best practices [19] [18].

### Blockchain Settlement Pilots Show Promise but Require Security Guardrails
Live DLT pilots in the EU (e.g., CSD Prague, 21X AG under the DLT Pilot Regime) and India's Vajra blockchain for UPI clearing are demonstrating the potential to reduce settlement times from days to near real-time [20] [21]. However, scaling these solutions for cross-border tokenized settlement requires addressing significant security risks, particularly around smart contract vulnerabilities and cryptographic key management. Standardizing Multi-Party Computation (MPC) for key custody and mandating oracle audits are critical next steps.

### Public-Private Partnerships Consistently Outperform Solo Efforts
Established PPPs demonstrate clear benefits in enhancing cyber resilience. Models like FS-ISAC, Italy's CERTFin, and the UK's FSCCC provide structured platforms for intelligence sharing and coordinated response [22] [23] [24]. The Euro Cyber Resilience Board's CIISI-EU initiative offers a strong blueprint for a proposed India-EU BFSI Cyber Council, which could formalize daily threat intelligence flows using STIX/TAXII and pool resources for red-teaming exercises [25].

### Crisis Communications Can Move Markets and Mitigate Panic
Major operational outages have proven to have immediate macrofinancial impacts. The **October 2020** TARGET2 incident, which lasted 11 hours, and the **February 2021** NSE trading halt both caused significant market disruption [CITE_NOT_FOUND][CITE_NOT_FOUND]. To prevent market panic and limit legal exposure during a cross-border cyber incident, it is essential to have pre-agreed "golden hour" disclosure scripts aligned with market abuse regulations like the EU's MAR Article 17 and SEBI's LODR Schedule III [26] [CITE_NOT_FOUND].

## 1. Threat Landscape 2023-2025: A Shared Battlefield

The financial sectors in both India and the European Union are facing an unprecedented and escalating wave of sophisticated cyber threats. While the specific volume and types of attacks may vary, the underlying trends point to a convergence in attacker capabilities and a shared vulnerability to systemic disruption.

### Dominant Threat Vectors: Ransomware, Supply Chain, and AI-Driven Attacks

For 2023-2025, the threat landscape for both regions is dominated by several high-impact vectors:

* **Ransomware:** Ransomware-as-a-Service (RaaS) has lowered the entry barrier for criminals, leading to a surge in attacks [27]. In India, ransomware incidents surged **53%** in 2022, and the country is a primary target for groups like KillSec, FunkSec, and RansomHub [CITE_NOT_FOUND][28]. These attacks have evolved beyond simple encryption to complex extortion tactics, including data theft and threats of physical sabotage [29] [6]. In the EU, ransomware accounted for **36%** of incidents in the finance sector, primarily affecting service providers (**29%**) and insurance organizations (**17%**), leading to financial loss, data exposure, and operational disruption [30] [31]. A notable 2024 incident in India involved a misconfigured Jenkins server at a TCS-SBI joint venture collaborator, which was exploited by a ransomware group, disrupting the banking ecosystem [29].

* **AI-Driven Attacks:** Cybercriminals are increasingly leveraging generative AI to create more sophisticated and adaptive threats [29]. This includes AI-powered malware like the BlackMamba keylogger, highly targeted phishing scams, and AI-generated voice and video deepfakes for executive fraud [6] [29]. The RBI has explicitly urged Indian financial institutions to adopt "AI-aware defense strategies" in response [18].

* **Supply Chain and Third-Party Risk:** The financial sector's heavy reliance on third-party vendors for everything from core banking software to cloud services creates a massive attack surface. The **January 2023** ransomware attack on ION Group, a UK-based software provider, disrupted derivatives trading for major banks in the US and Europe [7] [8]. In the EU, attacks on suppliers have led to the exposure of sensitive data in **63%** of cases [31]. In India, over **71%** of security leaders identify third-party vulnerabilities as a significant attack pathway [27].

* **Cloud and API Vulnerabilities:** The rapid migration to cloud services has introduced new risks from misconfigured environments and insecure APIs. In India, large organizations have an average of 12,000 internet-facing assets susceptible to exploitation [CITE_NOT_FOUND]. Cloud exploits emerged as a critical entry point in H1 2024, with attackers targeting weak IAM settings, publicly exposed storage, and API vulnerabilities to escalate privileges and exfiltrate data [CITE_NOT_FOUND][32].

* **Phishing and Social Engineering:** Phishing remains a primary initial access vector. India's BFSI sector accounts for nearly **30%** of all phishing attacks, with the banking sub-sector being the most targeted [28]. Indian organizations receive **50%** more suspicious emails than the global average [CITE_NOT_FOUND]. In the EU, social engineering campaigns affect both individuals (**38%**) and credit institutions (**36%**) [31].

### Geopolitically Motivated Attacks and DDoS

Geopolitical tensions are a significant driver of cyberattacks. Pro-Russian hacktivist groups like NoName057(16) and KillNet have launched waves of Distributed Denial of Service (DDoS) attacks against European financial institutions, including the European Investment Bank, in retaliation for support of Ukraine [33] [34] [35]. These attacks, while often limited in direct financial impact, cause operational disruption and serve as a tool for political destabilization [34]. India is the second-most targeted country for DDoS attacks globally, accounting for **13.22%** of total attacks [CITE_NOT_FOUND][27].

| Threat Vector | India BFSI Sector | EU Financial Sector |
| :--- | :--- | :--- |
| **Ransomware** | 96.6% of dark web threats against India target BFSI [36]. 53% surge in incidents (2022) [CITE_NOT_FOUND]. | 36% of incidents in the finance sector [30]. Primarily affects service providers (29%) and insurers (17%) [31]. |
| **Phishing** | BFSI sector accounts for nearly 30% of all phishing attacks [28]. | Prevalent tactic affecting individuals (38%) and credit institutions (36%) [31]. |
| **Supply Chain** | 71.9% of respondents see third-party vulnerabilities as a significant pathway [27]. | Attacks on suppliers led to data exposure (63%), operational disruption (26%), and financial loss (11%) [31]. |
| **DDoS Attacks** | 13.22% of total global DDoS attacks, 2nd highest volume [CITE_NOT_FOUND][27]. | 58% of DDoS incidents targeted European credit institutions [31]. |
| **AI-Driven Attacks** | Predicted to dominate the 2025 threat landscape; RBI mandates "AI-aware defense" [29] [18]. | Increase in cyber-enabled fraud, phishing, and social engineering attacks noted [37]. |
| **Data Breaches** | Average cost reached INR 179 million, a 28% increase since 2020 [CITE_NOT_FOUND]. | Finance industry has the highest average cost of a breach for 12 consecutive years [CITE_NOT_FOUND]. |

**Key Takeaway:** While specific attack volumes differ, the core threats of ransomware, supply chain compromise, and AI-weaponization are universal. This shared threat landscape creates a strong incentive for joint defense initiatives.

## 2. Recent High-Impact Incidents: Lessons from the Front Lines

A review of significant cyber incidents from 2023-2025 provides concrete evidence of the threats facing the financial sector and highlights critical control gaps. These events underscore the systemic nature of cyber risk, where a single vulnerability in a third-party provider can trigger cascading failures across the global financial system.

### Case Study 1: ION Group Ransomware Attack (January 2023)
* **Incident:** The LockBit ransomware gang attacked the Cleared Derivatives division of ION Group, a UK-based financial software firm [7].
* **Impact:** The attack caused significant disruption to derivatives trading in the US and Europe. Major clients, including ABN Amro Clearing and Intesa Sanpaolo, were forced to process trades manually, leading to significant delays [8] [9]. The U.S. Commodity Futures Trading Commission (CFTC) had to delay the publication of weekly trading statistics [8].
* **Root Cause:** Ransomware attack exploiting an unspecified vulnerability. LockBit claimed a ransom was paid, though ION has not commented [38].
* **Lesson Learned:** This incident is a stark illustration of **supply-chain risk** and the potential for a single point of failure in critical market infrastructure to cause widespread, cross-border disruption. It highlights the need for robust vendor oversight and resilient backup processing capabilities.

### Case Study 2: Snowflake Data Breach (May-June 2024)
* **Incident:** A massive data breach campaign targeted customers of cloud data platform Snowflake. Attackers gained access using credentials stolen via infostealer malware from customer systems that lacked multi-factor authentication (MFA) [CITE_NOT_FOUND].
* **Impact:** High-profile financial institutions were affected. **Santander Bank** confirmed a breach affecting data of 30 million customers and staff, including bank account details and credit card numbers [39]. Data for over 12,000 US-based Santander employees was also leaked [40].
* **Root Cause:** Compromised customer credentials combined with a lack of MFA. Snowflake stated its own core systems were not breached [CITE_NOT_FOUND].
* **Lesson Learned:** This incident underscores the critical importance of **identity and access management (IAM)**. It demonstrates that even with a secure cloud platform, weak customer-side controls (single-factor authentication) create a catastrophic vulnerability. Mandatory MFA and continuous monitoring for credential compromise are non-negotiable controls.

### Case Study 3: Capita Cyber Security Breach (March 2023)
* **Incident:** UK-based outsourcing firm Capita, which provides pension administration services, suffered a ransomware attack.
* **Impact:** The personal data of **6.6 million** people was stolen, including sensitive financial and criminal record data for some individuals [41]. Over 325 pension schemes were impacted [CITE_NOT_FOUND].
* **Root Cause:** A malicious file was downloaded, and despite a security alert, the device was not quarantined for **58 hours**, allowing the attacker to escalate privileges and move laterally across the network [41].
* **Regulatory Response:** The UK's Information Commissioner's Office (ICO) imposed a **£14 million** fine on Capita for failing to implement appropriate security measures, citing failures in preventing privilege escalation, inadequate response to alerts, and insufficient penetration testing [42] [41].
* **Lesson Learned:** This case highlights the severe financial and reputational consequences of failing to adhere to basic cyber hygiene and incident response protocols. A slow response to an initial alert turned a containable breach into a massive data exfiltration event.

### Case Study 4: Pro-Russian DDoS Campaigns (2023)
* **Incident:** Pro-Russian hacktivist groups, including KillNet and NoName057(16), launched a series of DDoS attacks against European financial institutions.
* **Impact:** The **European Investment Bank (EIB)** suffered severe website availability problems in June 2023 [43]. Similar attacks disrupted online banking access for banks in the Czech Republic and Poland in August 2023 [33].
* **Root Cause:** Politically motivated attacks in retaliation for European support for Ukraine [35].
* **Lesson Learned:** Geopolitical events are a direct catalyst for cyberattacks on the financial sector. Institutions must have robust DDoS mitigation strategies and be prepared for politically motivated disruption, which can impact customer confidence and service availability.

### Case Study 5: CDSL KRA Cyber Incident (October 2023)
* **Incident:** A vulnerability was reported in the KYC Registration Agency (KRA) system of India's Central Depository Services Limited (CDSL).
* **Impact:** The incident reportedly exposed the personal and financial data of millions of Indian investors. As a precautionary measure, CDSL Ventures temporarily shut down the system, halting the onboarding of new clients for major stockbrokers like Zerodha and Groww [CITE_NOT_FOUND].
* **Root Cause:** Unspecified vulnerability in the KRA system.
* **Lesson Learned:** This incident highlights the systemic importance of market infrastructure entities like depositories. A vulnerability in a central KYC database can have immediate, sector-wide consequences, disrupting market operations and exposing massive amounts of sensitive data. It reinforces the need for rigorous security testing and resilience planning for all Financial Market Infrastructures (FMIs).

## 3. Regulatory Convergence Snapshot: DORA and CERT-In Raise the Bar

As of late 2025, the regulatory landscapes in both the EU and India have undergone significant transformations, creating a more stringent but complex compliance environment for financial institutions. The EU's Digital Operational Resilience Act (DORA) became fully applicable on **January 17, 2025**, while India has seen a wave of new mandates from the RBI and CERT-In, most notably the **CERT-In Guidelines 2025** and the RBI's push towards Zero Trust Architecture [3] [13] [17].

### The EU's DORA Framework: A Harmonized Approach
DORA (Regulation (EU) 2022/2554) establishes a unified and comprehensive framework for digital operational resilience across 20 different types of EU financial entities [3] [44]. It is considered *lex specialis*, meaning its specific rules for the financial sector supersede the more general requirements of the NIS2 Directive in areas like ICT risk management and incident reporting [45] [46].

DORA is built on five key pillars:
1. **ICT Risk Management:** Mandates a comprehensive and well-documented ICT risk management framework, with the management body holding ultimate responsibility [47].
2. **Incident Reporting:** Establishes a harmonized, multi-stage reporting process for major ICT-related incidents: an initial notification within **4 hours** of classification (and no later than 24 hours from awareness), an intermediate report within **72 hours**, and a final report within **one month** [1] [2].
3. **Digital Operational Resilience Testing:** Requires a program of regular testing, including advanced Threat-Led Penetration Testing (TLPT) at least every **three years** for significant entities, based on the TIBER-EU framework [48] [CITE_NOT_FOUND].
4. **ICT Third-Party Risk Management:** Imposes strict rules for managing risks from ICT service providers, including detailed contractual requirements for access, audit, and exit strategies [49].
5. **Oversight of Critical ICT Third-Party Providers (CTPPs):** Creates a pan-European oversight framework for providers deemed critical to the financial system, led by the European Supervisory Authorities (ESAs) [3].

### India's Multi-Pronged Regulatory Push
India's regulatory approach is characterized by a series of directives from multiple bodies, creating a layered and demanding compliance environment.

* **RBI's 2025 Mandates (Zero Trust & Resilience):** The Reserve Bank of India is driving a paradigm shift from perimeter-based security to a **Zero Trust Architecture (ZTA)** [17]. This is no longer an aspirational goal but a regulatory expectation, emphasizing principles like continuous verification, identity-first security, least privilege access, and micro-segmentation [17]. The RBI's **Master Direction on IT Governance, Risk, Controls and Assurance Practices** (effective April 1, 2024) and **Master Direction on Outsourcing of IT Services** (2023) provide the foundational rules for governance, risk assessment, and third-party oversight [50] [51].

* **CERT-In's Expanded Mandates:** The **CERT-In Directions of April 2022** established a strict **6-hour** window for reporting a wide range of cyber incidents and mandated the retention of logs within Indian jurisdiction for 180 days [4] [CITE_NOT_FOUND]. The new **CERT-In Guidelines 2025** (effective July 25, 2025) expand this scope to all businesses, mandating annual third-party audits and comprehensive Software Bill of Materials (SBOM) requirements, explicitly holding contracting entities responsible for their supply chain's security [13].

* **SEBI and IRDAI:** Other sectoral regulators are aligning with this trend. SEBI's **Cyber Security and Cyber Resilience Framework (CSCRF)** for regulated entities and IRDAI's **Information and Cybersecurity Guidelines 2023** establish specific standards for their respective sectors, including requirements for threat intelligence, SOC operations, and VAPT [CITE_NOT_FOUND][27].

### The Compliance Challenge: Navigating Dual Regimes
For financial institutions operating in both India and the EU, the primary challenge lies in reconciling these overlapping but distinct regulatory frameworks. The differing incident reporting timelines—DORA's multi-stage process versus CERT-In's 6-hour rule—create immediate operational friction. Without a harmonized approach, firms face the burden of managing multiple reporting clocks, formats, and taxonomies, increasing compliance costs and the risk of penalties.

## 4. Compliance Heat-Map: Critical Gaps and Quick Wins

For financial institutions operating across India and the EU, navigating the dual regulatory landscapes of DORA and the RBI/CERT-In mandates presents a significant challenge. While both frameworks share the goal of enhancing cyber resilience, their specific requirements diverge in critical areas. The following heat-map identifies key control domains, maps the respective regulations, highlights alignment gaps, and proposes "quick-win" solutions to bridge them ahead of the dialogue.

| Control Domain | EU Framework (DORA) | Indian Framework (RBI/CERT-In) | Alignment Gap & Risk | Quick-Win Convergence Action |
| :--- | :--- | :--- | :--- | :--- |
| **Governance & Accountability** | Management body holds ultimate responsibility for ICT risk. Clear roles for risk, control, and audit functions [47]. | Board-level IT Strategy Committee and senior CISO role mandated. CISO must be independent of IT and report to risk management [52]. | **Low (Green):** Both frameworks mandate top-level ownership and clear CISO responsibility. The structures are highly compatible. | **Harmonize CISO reporting lines.** Adopt the stricter Indian model of CISO independence from IT across all operations to ensure a unified governance posture. |
| **ICT Risk Management** | Requires a comprehensive, documented ICT risk management framework, reviewed at least annually. Includes identification, protection, detection, response, and recovery [47] [53]. | Mandates an IT and Information Security Risk Management Framework, Cyber Security Policy, and Cyber Crisis Management Plan (CCMP) [52]. | **Low (Green):** Both require a holistic, documented risk management approach. The core principles are strongly aligned. | **Create a unified risk taxonomy.** Map DORA's framework elements to RBI's policy requirements to create a single, cross-referenced master policy document. |
| **Incident Reporting** | **Multi-stage:** Initial report (4h post-classification/24h post-awareness), intermediate (72h), final (1 month) [1] [2]. | **Single, rapid stage:** Report specified incidents to CERT-In within **6 hours** of notice. RBI also requires reporting within **2-6 hours** for banks [4] [54]. | **High (Red):** Conflicting timelines create significant operational friction and risk of non-compliance with the stricter 6-hour rule. | **Adopt a "report-once, distribute-many" model.** Develop an internal "zero-hour" process that triggers parallel, automated notifications to both CERT-In and EU authorities, using a standardized template that meets both frameworks' data requirements. |
| **Resilience Testing** | **Threat-Led Penetration Testing (TLPT)** at least every **3 years** for significant entities, based on TIBER-EU framework [48] [CITE_NOT_FOUND]. | **Annual Penetration Testing (PT)** and **bi-annual Vulnerability Assessment (VA)** for critical systems. Half-yearly DR drills for critical systems [27] [52]. | **Medium (Amber):** Indian mandates are more frequent (annual PT vs. 3-yearly TLPT), but EU's TLPT is more sophisticated (intelligence-led). Risk of "testing for the test" vs. real-world resilience. | **Establish a "Joint Test Calendar."** Conduct the annual Indian-mandated PT and use its findings to inform the threat intelligence scenarios for the triennial EU-mandated TLPT, creating a continuous improvement loop. |
| **Third-Party & Cloud Oversight** | **CTPP Oversight Framework:** Direct oversight of critical providers by ESAs. Mandates unrestricted audit/access rights and detailed exit/portability clauses in contracts [3] [55]. | **Master Directions on Outsourcing:** REs remain fully responsible. Mandates unrestricted audit/access for RE and RBI, even cross-border. Requires tested exit strategies and robust BCP/DR for vendors [51]. | **Medium (Amber):** EU's direct oversight of CTPPs is a major structural difference. Indian firms relying on EU-designated CTPPs may have unclear lines of supervisory authority. | **Draft a "Joint Supervisory Protocol."** Propose a protocol for the dialogue that outlines how RBI can leverage the findings from ESA-led CTPP audits for Indian REs, avoiding duplicative and costly vendor audits. |
| **Zero Trust Architecture (ZTA)** | Not explicitly mandated, but principles are embedded in risk management and security requirements. | **Explicitly mandated** by RBI as a core principle for 2025, emphasizing identity-first security, least privilege, and micro-segmentation [17] [56]. | **Low (Green):** India's explicit mandate provides a clear, high standard. EU firms are moving in this direction organically. This is an area of strong potential alignment. | **Co-sponsor a ZTA pilot.** Launch a joint pilot for implementing ZTA in a cross-border payment system, using NIST SP 800-207 as a common reference architecture. |

**Key Takeaway:** The most critical and immediate gap is in incident reporting timelines. A harmonized "report-once" model is the highest-priority quick win. For longer-term convergence, focusing on joint supervisory protocols for third-party risk and co-sponsoring pilots on Zero Trust offers the most strategic value.

## 5. Data-Sharing & Threat Intelligence Cooperation: A TLP-RED Lane

Effective cybersecurity relies on the timely sharing of threat intelligence, but this is often hindered by conflicting legal and data protection frameworks. For India and the EU, the primary challenge lies in reconciling the EU's GDPR, which prioritizes data protection, with India's dual requirements of the DPDP Act 2023 and the IT Act, which mandate rapid security reporting [57] [4]. A structured, trust-based operating model is essential to enable real-time cooperation.

### Legal Guardrails for Cross-Border Data Sharing

* **EU GDPR:** Cross-border data transfers are strictly regulated. As India does not have an "adequacy decision" from the EU, transfers must rely on **Standard Contractual Clauses (SCCs)**, which require data exporters to conduct a Transfer Impact Assessment (TIA) to ensure the recipient country's laws do not undermine GDPR protections [58] [CITE_NOT_FOUND]. However, GDPR's **Recital 49** recognizes that processing personal data for network and information security can constitute a "legitimate interest," providing a potential legal basis for sharing cybersecurity telemetry, provided it is strictly necessary and proportionate [59].

* **India's DPDP Act 2023:** India employs a "blacklist" approach, permitting data transfers to any country not specifically restricted by the government [60]. However, Section 16(2) of the Act preserves stricter sectoral laws, such as the **RBI's 2018 payment data localization mandate**, which requires all payment system data to be stored only in India [61] [62]. Furthermore, the **CERT-In Directions of 2022** require logs to be maintained within Indian jurisdiction for 180 days [4]. These localization requirements create significant friction for sharing raw telemetry data.

### A Proposed Operating Model for Threat Intelligence Sharing

To navigate these complexities, a dedicated operating model for threat intelligence (TI) sharing between CERT-In, CERT-EU, ENISA, and FS-ISAC is proposed. This model would be governed by a formal Memorandum of Understanding (MoU) and built on principles of trust, standardization, and privacy-by-design.

**1. Governance and Trust Framework:**
* **Founding MoU:** A formal MoU should establish the legal basis, scope, and rules of engagement. It should reference GDPR's legitimate interest for security and India's IT Act Section 70B, which mandates CERT-In's role in information sharing [63].
* **Trusted Community:** The partnership should be structured as a trusted community, leveraging existing vetting models like TF-CSIRT's Trusted Introducer to establish confidence among participants [CITE_NOT_FOUND].
* **Clear Roles:**
 * **CERT-In & CERT-EU:** Act as national/supranational hubs for receiving, sanitizing, and disseminating intelligence.
 * **ENISA:** Provides strategic analysis, good practices, and supports the CSIRTs Network [64].
 * **FS-ISAC:** Acts as the primary private-sector hub, aggregating anonymized intelligence from member firms and feeding it into the government channel [22].

**2. Technical Interoperability and Data Handling:**
* **Standardized Formats:** Mandate the use of **STIX 2.1** for expressing threat intelligence and **TAXII 2.1** for transport, ensuring machine-readable and automated exchange [65] [66]. MISP should be used as the core open-source platform, leveraging its existing STIX/TAXII integration capabilities [67].
* **Traffic Light Protocol (TLP) 2.0:** All shared intelligence must be marked according to TLP 2.0. A dedicated **TLP:RED** channel should be established for highly sensitive, need-to-know information shared only between the designated CERTs for immediate action [68].
* **Privacy-by-Design Redaction:** Before sharing, all data must be passed through an automated "scrubbing" process to remove or pseudonymize Personally Identifiable Information (PII) and other sensitive data not strictly necessary for security analysis. This aligns with GDPR's data minimization principle and guidance from CISA's Automated Indicator Sharing (AIS) program [CITE_NOT_FOUND][CITE_NOT_FOUND].

**3. Ingestion and Dissemination Workflow:**
1. **Submission:** An Indian or EU financial firm detects a threat and reports it to FS-ISAC (private sector) and its national regulator/CERT (public sector).
2. **Triage & Enrichment (FS-ISAC):** FS-ISAC's Global Intelligence Office (GIO) triages, enriches, and anonymizes the submission, then shares it with members via its platforms (Connect/Share) and with government partners via the TLP:RED channel [69].
3. **Sanitization & Correlation (CERTs):** CERT-In and CERT-EU receive intelligence from their respective constituencies and from FS-ISAC. They use automated tools (like MISP) to correlate the data, sanitize it for PII, and apply appropriate TLP markings [67].
4. **Actionable Dissemination:** The CERTs disseminate actionable alerts and IoCs to their respective financial sectors. For cross-border threats, CERT-In and CERT-EU exchange the sanitized, TLP-marked intelligence directly via the secure TAXII channel.
5. **Strategic Analysis (ENISA):** ENISA receives aggregated, anonymized data to produce strategic threat landscape reports and identify cross-sectoral vulnerabilities [70].

This federated model respects data localization rules by sharing sanitized, actionable intelligence rather than raw data, while the strict use of TLP and a trusted-partner framework provides the legal and operational confidence needed for real-time cooperation.

## 6. Emerging Technologies & Solutions: Pilots for a Resilient Future

The BFSI sector is increasingly turning to emerging technologies to counter sophisticated cyber threats. AI-driven security platforms, Zero Trust Architecture, and Distributed Ledger Technology (DLT) are moving from hype to reality, with regulators in both India and the EU actively encouraging their adoption. Co-sponsoring targeted pilots in these areas can accelerate learning, validate ROI, and build a common foundation for future standards.

### Technology Readiness in 2025

* **AI-Driven XDR/SOAR:** The market for Extended Detection and Response (XDR) and Security Orchestration, Automation, and Response (SOAR) is maturing rapidly. Forrester's Q2 2024 Wave report notes that XDR is now a viable replacement for SIEM in many use cases [71]. Leading platforms from vendors like Microsoft, Palo Alto Networks, and CrowdStrike are integrating Generative AI to automate incident summaries, translate query languages, and even deploy AI agents for response [72]. Financial institutions using AI-driven security report **28% faster** Mean Time to Respond (MTTR) and significantly lower data breach costs [CITE_NOT_FOUND]. The RBI is explicitly pushing for "AI-aware defense strategies" [18].

* **Zero Trust Enforcement Stacks:** Zero Trust is no longer a theoretical concept but a regulatory mandate in India and a strategic priority in the EU [17] [CITE_NOT_found]. Based on the principle of "never trust, always verify," ZTA relies on a stack of enforcement technologies, including identity-centric controls, micro-segmentation, and continuous monitoring. NIST SP 800-207 provides a vendor-agnostic reference architecture, and case studies from vendors like Illumio and Zscaler demonstrate measurable benefits in containing ransomware and reducing attacker dwell time [73] [CITE_NOT_FOUND].

* **Attack Surface Management (ASM):** ASM, which includes External Attack Surface Management (EASM) and Cyber Asset Attack Surface Management (CAASM), is a cornerstone of modern threat management [CITE_NOT_FOUND]. Analyst firms like Gartner and Forrester have established market guides, and platforms from vendors like Veriti, Censys, and Microsoft are providing automated, real-time insights into exposures, enabling proactive mitigation [74] [CITE_NOT_FOUND].

* **Deception Technology:** Deception platforms, which use honeypots and decoy accounts to mislead and detect attackers, are now considered an essential component of a proactive defense, especially when integrated into a Zero Trust architecture [75] [76]. MITRE's D3FEND framework codifies deception as a core defensive tactic [77].

* **Distributed Ledger Technology (DLT):** DLT is moving from experiment to production. The **EU's DLT Pilot Regime** (Regulation (EU) 2022/858) is live, with authorized infrastructures like CSD Prague and 21X AG settling transactions in tokenized instruments [78] [20]. In India, the **NPCI's Vajra platform** uses a permissioned blockchain for payment clearing, and the **RBI's CBDC pilots** are exploring both wholesale and retail use cases [21] [79]. While promising for settlement integrity and fraud prevention, key risks around smart contract security, oracle manipulation, and cryptographic key management must be addressed.

### Proposed India-EU Pilot Programs

To accelerate safe adoption and foster interoperability, India and the EU should co-sponsor three strategic pilots. These pilots must be governed by the stringent requirements of the **EU AI Act** for high-risk systems (which includes finance) and India's **DPDP Act** and **RBI mandates**, ensuring data protection, transparency, and human oversight [19] [57].

**Pilot 1: AI-Assisted Cross-Border SOC**
* **Objective:** To validate the effectiveness of an AI-driven SOAR platform in a joint Security Operations Center (SOC) environment for reducing Mean Time to Detect (MTTD) and Mean Time to Respond (MTTR) for cross-border financial threats.
* **Architecture:** A federated model where a shared, vendor-agnostic SOAR platform (using open standards like OCSF) ingests alerts from the respective SOCs of participating Indian and EU banks. An AI agent, governed by the EU AI Act's logging and transparency rules, would correlate alerts, propose response playbooks, and automate initial containment actions.
* **Success Metrics:**
 * Reduce combined MTTD for cross-border incidents by **30%** from baseline.
 * Reduce combined MTTR by **40%**.
 * Achieve **95%** accuracy in AI-proposed incident categorization.
* **Timeline:** 12 months (Q1 2026 - Q4 2026).

**Pilot 2: Zero Trust for Cross-Border Payments**
* **Objective:** To implement and test a Zero Trust Architecture for a critical cross-border payment messaging system (e.g., a SWIFT alternative or overlay) to prevent lateral movement and data exfiltration.
* **Architecture:** Based on NIST SP 800-207, the pilot would use micro-segmentation to isolate payment gateways, continuous identity-based authentication for all API calls, and a policy enforcement point to validate every transaction request.
* **Success Metrics:**
 * Demonstrate **100%** containment of a simulated breach within a single micro-segment during red team testing.
 * Reduce credential-based access risk by **90%** through enforcement of least-privilege policies.
 * Maintain **99.99%** uptime and transaction processing latency within agreed SLAs.
* **Timeline:** 18 months (Q1 2026 - Q2 2027).

**Pilot 3: DLT for Trade Finance & KYC Attestation**
* **Objective:** To test a DLT platform for securing trade finance documentation and providing privacy-preserving KYC attestations, reducing fraud and operational overhead.
* **Architecture:** A permissioned DLT (e.g., based on Hyperledger or Corda) where trade documents (Bills of Lading, Letters of Credit) are recorded as immutable assets. The platform would integrate with the EU's EUDI Wallet and India's Account Aggregator framework, allowing parties to provide zero-knowledge proofs of identity and creditworthiness without sharing underlying data. Security would be enhanced with MPC for key management.
* **Success Metrics:**
 * Reduce trade finance document fraud by **50%** in the pilot environment.
 * Decrease KYC/AML verification time from days to minutes.
 * Successfully demonstrate interoperability between EUDI Wallet and Account Aggregator consent mechanisms.
* **Timeline:** 18 months (Q2 2026 - Q4 2027).

## 7. Cloud & CTPP Oversight: Managing Concentration Risk

The financial sector's heavy reliance on a small number of large Cloud Service Providers (CSPs) has created a significant concentration risk, a concern shared by regulators in both India and the EU. Both DORA and RBI's outsourcing directives have introduced stringent measures to manage this systemic vulnerability, focusing on resilience, auditability, and viable exit strategies.

### The Regulatory Response to Concentration Risk

* **EU's DORA Framework:** DORA establishes a pan-European oversight framework for **Critical ICT Third-Party Providers (CTPPs)**, which will include major CSPs [3]. Under this framework, the European Supervisory Authorities (ESAs) act as Lead Overseers, with direct powers to request information, conduct inspections (including on-site), and issue recommendations to CTPPs [80]. DORA's **Article 30** mandates that contracts with ICT providers include unrestricted rights of access, inspection, and audit for financial entities and their regulators, especially for critical functions [81]. It also requires comprehensive and tested exit strategies to ensure data portability and service continuity [44].

* **India's RBI Mandates:** The **RBI's Master Direction on Outsourcing of IT Services (2023)** explicitly requires Regulated Entities (REs) to assess and manage concentration risk [51]. Like DORA, it mandates that contracts grant unrestricted access and audit rights to the RE and to the RBI, even for providers in foreign jurisdictions. The directive places a strong emphasis on business continuity, requiring service providers to have robust, tested BCP/DRP frameworks. The appendix on cloud services specifically details requirements for exit strategies, including data portability, secure data purging, and smooth transition to an alternative provider or in-house solution [51].

### Shared Challenges and Technical Solutions

Both regulatory frameworks aim to solve the same core problems: lack of visibility into provider operations, difficulty in switching providers (vendor lock-in), and ensuring service resilience during an outage.

**1. Enhancing Portability and Mitigating Vendor Lock-in:**
* **Technical Portability Patterns:** True multi-cloud portability can be enhanced by adopting vendor-agnostic technologies. **Kubernetes** provides a degree of application portability, though challenges remain with managed services, storage, and identity management. **Infrastructure-as-Code (IaC)** tools like Terraform and Crossplane can define infrastructure in a cloud-agnostic way, but require careful management of provider-specific modules [CITE_NOT_FOUND].
* **EU Data Act:** The EU's **Data Act (Regulation (EU) 2023/2854)** will be a game-changer. It mandates the elimination of cloud switching charges, including data egress fees, from **January 12, 2027** [82]. This will significantly reduce the financial barriers to exit and promote a more competitive cloud market.

**2. Ensuring Resilience and Testing Exit Strategies:**
* **Resilience Targets:** Both frameworks require resilience, but the RBI is more prescriptive in some areas. A 2025 RFP from the Central Bank of India, for example, specified a **Recovery Time Objective (RTO) of 15 minutes** and a **Recovery Point Objective (RPO) of 0 minutes** for a critical cash management service [16]. The RBI's IT Governance direction also mandates half-yearly DR drills for critical systems [83]. These provide concrete targets that can be adopted as a shared benchmark.
* **Exit Testing:** Both DORA and the RBI mandate that exit strategies be tested. This should involve annual tabletop exercises and biennial live-drill "war games" where a financial institution simulates a full migration of a critical application from one CSP to another, measuring the time, cost, and data loss.

### A Joint Supervisory Protocol

To address the oversight challenge, India and the EU should develop a joint supervisory protocol for CTPPs.

* **Shared Risk Register:** A common, shared risk register for CTPPs should be established, based on the template from DORA's **Article 28** register of information [84]. This would allow RBI and the ESAs to have a unified view of dependencies, concentration levels, and identified risks.
* **Cross-Audit Coordination:** The protocol should formalize **pooled audits**, as permitted by the RBI and EBA guidelines [51] [CITE_NOT_FOUND]. For a CTPP designated by the ESAs, the RBI could participate in the Joint Examination Team (JET) as an observer. The findings of the ESA-led audit would then be formally shared with the RBI, satisfying Indian supervisory requirements and avoiding duplicative, costly audits for both the provider and the financial institutions.
* **Joint Stress Scenarios:** India and the EU should co-develop severe-but-plausible stress scenarios for cloud concentration risk. An example scenario could be a simultaneous, multi-region outage of a major CSP, testing the ability of financial institutions in both jurisdictions to failover to alternative providers or in-house systems within the mandated RTO/RPO targets.

## 8. Capacity-Building & Skills: Closing a Critical 800,000-Person Gap

A severe and growing cybersecurity skills gap poses a direct threat to the stability of the financial sectors in both India and the EU. The EU faces a shortage of approximately **299,000** cybersecurity professionals, a figure that has risen **9%** since 2023, while India needs one million professionals but currently has less than half that number [11] [12]. This combined deficit of nearly 800,000 skilled workers will critically undermine the ability of financial institutions to comply with new regulations and defend against increasingly sophisticated threats.

### A Common Language for Cybersecurity Roles

To effectively address the skills gap, a common understanding of required roles and competencies is essential. Both the EU and the US have developed robust frameworks that can be adopted jointly:

* **ENISA's European Cybersecurity Skills Framework (ECSF):** This framework outlines 12 key professional profiles, detailing their missions, tasks, skills, and knowledge. It is a practical tool designed to bridge the gap between industry needs and educational programs [85].
* **NIST's NICE Workforce Framework:** This resource provides a granular taxonomy of Work Roles, Competency Areas, and specific Task, Knowledge, and Skill (TKS) statements, allowing for precise job role definition and curriculum development [86].

For the financial sector, four roles are of critical importance and should be the focus of joint capacity-building efforts:
1. **Cyber Threat Intelligence Specialist:** Collects, analyzes, and disseminates intelligence on threat actors and TTPs (ECSF Profile) [87].
2. **Cloud Security Architect:** Designs and develops secure cloud architectures (ECSF Profile) [87].
3. **Penetration Tester / Red Team Operator:** Assesses security effectiveness by simulating real-world attacks (ECSF Profile) [87].
4. **Cyber Incident Responder:** Monitors for, analyzes, and responds to cyber incidents (ECSF Profile) [87].

### A 12-Month Joint Curriculum Plan: The "Cyber Fellowship 500"

A scalable, high-impact solution is to create a joint "Cyber Fellowship 500" program, aiming to train and certify 500 professionals (250 from each region) over 12-18 months in these critical roles. This program would be a public-private partnership involving regulators (RBI, ESAs), national CERTs (CERT-In, CERT-EU), and private sector training partners.

**Curriculum Structure (12-Month Plan):**

* **Months 1-3: Foundational Skills (Common Track):**
 * **Target Cohort:** Entry-level professionals, career changers.
 * **Content:** Based on certifications like CompTIA Security+ and ISC2's Certified in Cybersecurity (CC). Covers core concepts of networking, security operations, and governance.
 * **Delivery:** Blended learning (online modules + weekly virtual labs).
 * **KPI:** **90%** of cohort achieves foundational certification.

* **Months 4-9: Specialization Tracks (Role-Specific):**
 * **Target Cohort:** Fellows choose one of the four critical role tracks.
 * **Content & Certifications:**
 * **Threat Intelligence:** GIAC Cyber Threat Intelligence (GCTI).
 * **Cloud Security:** AWS/Azure/GCP Security Specialty certifications.
 * **Red Teaming:** Offensive Security Certified Professional (OSCP), CREST CRT.
 * **Incident Response:** GIAC Certified Incident Handler (GCIH).
 * **Delivery:** Intensive, hands-on training using shared **cyber range** platforms. Both India (NCIIPC National Cyber Range, C3iHub IITK) and the EU (ENISA-supported ranges) have existing infrastructure that can be leveraged [CITE_NOT_FOUND][CITE_NOT_FOUND].
 * **KPI:** **80%** of fellows in each track pass their respective professional certification exams.

* **Months 10-12: Capstone Exercise & Placement:**
 * **Content:** A joint, cross-border cyber exercise simulating a supply-chain attack on a financial institution. Indian and EU fellows work together in a virtual SOC.
 * **Delivery:** "Train-the-trainer" model, where senior experts from CERT-In and ENISA lead the exercise, building internal capacity for future training [CITE_NOT_FOUND].
 * **Placement:** Fellows are placed in internships or full-time roles within participating financial institutions.
 * **KPI:** **95%** of graduating fellows placed in relevant cybersecurity roles within 3 months of program completion.

**Funding and Collaboration:**
* **Funding Sources:** The program can be funded through a combination of sources, including the EU's **Digital Europe Programme** (which has a €375 million budget for cybersecurity) and **Horizon Europe**, and Indian initiatives from **MeitY** and the **National Skill Development Corporation (NSDC)** [88] [CITE_NOT_FOUND]. Bilateral funds through the **India-EU Trade and Technology Council (TTC)** should also be explored [CITE_NOT_FOUND].
* **Collaboration Framework:** The program would be governed by a joint steering committee with representation from the **RBI, ESAs, CERT-In, ENISA, and Europol's EC3**. This aligns with the existing mandates of these organizations to promote capacity-building and international cooperation [89] [90] [91].

## 9. Public-Private Partnership Models: An India-EU BFSI Cyber Council

Effective cyber resilience in the financial sector cannot be achieved by governments or private firms acting alone. Public-Private Partnerships (PPPs) are essential for creating the trusted channels needed for rapid threat intelligence sharing, coordinated incident response, and joint capability building. Drawing lessons from successful models like FS-ISAC and the EU's CIISI-EU, a dedicated **India-EU BFSI Cyber Council** should be established as a key deliverable of the dialogue.

### Lessons from Existing PPP Models

Several established PPPs provide a blueprint for the structure and function of a joint council.

| PPP Model | Governance & Membership | Key Functions | Funding Model |
| :--- | :--- | :--- | :--- |
| **FS-ISAC (Global)** | Member-driven non-profit. Board of cybersecurity executives from ~5,000 member firms across 75 countries. Membership limited to regulated financial institutions [22] [69]. | Real-time intelligence sharing via dedicated platforms (Share, Connect). Formal liaisons with government agencies (CISA, NCSC, Europol). Incident response coordination and exercises [69] [92]. | Member fees, tiered by institution size (assets/revenue) [CITE_NOT_FOUND]. |
| **CIISI-EU (EU)** | Market-driven initiative catalyzed by the ECB. Members include pan-European FMIs, central banks (in operational capacity), critical service providers, ENISA, and Europol. Governed by a non-binding Rulebook [25] [93]. | Facilitates sharing of tactical and strategic intelligence within a "trusted circle." Authorities (regulators/supervisors) are not members to maintain a non-regulatory space [94]. | Members contribute to costs, including for third-party TI and platform providers (e.g., Security Alliance, CIRCL MISP) [93]. |
| **CERTFin (Italy)** | Jointly led by the Bank of Italy and the Italian Banking Association (ABI), operated by ABI Lab. Strategic Committee sets policy, Steering Committee manages operations [23]. | Acts as a Single Point of Contact (PoC) for the Italian financial sector. Facilitates information exchange, supports incident response, and develops sector-specific guidelines [23]. | Jointly funded by the central bank and the banking association [CITE_NOT_FOUND]. |
| **FSCCC (UK)** | Partnership between NCSC, financial authorities (BoE, FCA), and industry (UK Finance). Coordinated by a Fusion Cell hosted at the NCSC and enabled by the i100 scheme [24]. | Identifies, investigates, and coordinates response to incidents with potential systemic consequences. Combines and analyzes intelligence from public and private sources [24]. | Jointly funded by government and industry contributions [CITE_NOT_FOUND]. |

**Key Takeaway:** Successful models combine a clear governance charter, a trusted membership circle, formal data-sharing protocols (like TLP), and a clear separation between voluntary intelligence sharing and mandatory regulatory reporting.

### Proposed Structure for an India-EU BFSI Cyber Council

The India-EU BFSI Cyber Council should be modeled on the CIISI-EU framework, creating a trusted, operational-level forum for public and private sector experts.

**Draft Constitution & Charter:**

* **Article 1: Mission:** To enhance the collective cyber resilience of the Indian and European Union financial sectors by facilitating the timely and actionable exchange of threat intelligence and promoting joint preparedness and response activities.
* **Article 2: Membership:**
 * **Core Members:** A limited number of systemically important banks and financial market infrastructures (FMIs) from both India and the EU.
 * **Public Sector Participants:** CERT-In, CERT-EU, NCIIPC, and the operational (non-supervisory) arms of the RBI and ECB.
 * **Advisory Participants:** ENISA and Europol's EC3.
* **Article 3: Governance:**
 * The Council will be co-chaired by representatives from the Indian Banks' Association (IBA) and the European Banking Federation (EBF).
 * A rotating **Secretariat**, managed jointly by FS-ISAC and India's CSIRT-Fin, will handle administrative and operational coordination.
 * Decisions will be made by consensus. A formal, non-binding **Rulebook** based on the CIISI-EU model will govern all activities [93].
* **Article 4: Data Sharing & Legal Framework:**
 * All information sharing will adhere strictly to the **Traffic Light Protocol (TLP) 2.0** [68].
 * The legal basis for sharing will be GDPR Recital 49 (network and information security) and India's IT Act Section 70B (CERT-In's mandate) [59] [91].
 * A **Safe Harbor** provision, aligned with EU NIS2 Directive's voluntary sharing protections, will shield members from liability for information shared in good faith within the Council [CITE_NOT_FOUND].
 * Data will be exchanged in **STIX 2.1** format via a shared **MISP** instance to ensure technical interoperability [65] [67].
* **Article 5: Funding:** The Council will be funded through a combination of annual member dues and seed funding from bilateral India-EU cooperation funds.
* **Article 6: Success Metrics:** The Council's effectiveness will be measured by KPIs including: time-to-share for critical IoCs, number of members contributing intelligence, and reduction in incident recovery time based on shared intelligence.

**90-Day Standing Agenda for the Council:**

* **Weekly Tactical Threat Brief (Mondays, TLP:AMBER):**
 * Review of top 5 active campaigns targeting BFSI in each region.
 * Sharing of new, anonymized IoCs.
 * Updates on major vulnerability patching.
* **Monthly Operational Sync (First Thursdays, TLP:AMBER):**
 * Deep dive on 1-2 significant incidents from the past month (root cause, TTPs).
 * Review of TI quality and platform effectiveness metrics.
 * Planning for upcoming joint exercises (e.g., phishing simulation, DDoS drill).
* **Quarterly Strategic Review (Last Fridays, TLP:GREEN):**
 * Presentation of threat landscape trends to senior leadership and regulatory observers.
 * Review of progress against the regulatory convergence roadmap.
 * Approval of budget and priorities for the next quarter.

## 10. Preparatory Meeting Agenda (November 4, 2025): Securing a Mandate for Action

The preparatory meeting on **November 4, 2025**, is the critical window to secure consensus on a concrete and ambitious work plan. The goal is not to debate the problems, but to finalize a set of actionable deliverables for the main dialogue on December 4. The agenda should be decision-oriented, with clear owners and pre-vetted draft texts ready for endorsement.

**Meeting Objective:** To achieve consensus on the draft texts of three core deliverables: a Joint Statement, a Regulatory Convergence Roadmap, and a Threat Intelligence Sharing Framework.

**Participants:**
* **Co-Chairs:** Secretary, Department of Economic Affairs (India); Director-General, DG ECFIN (EU Commission).
* **Workstream Owners (India):** Joint Secretary (Cyber Diplomacy, MEA), Head of CSIRT-Fin, CISO (RBI).
* **Workstream Owners (EU):** Head of Unit (Cybersecurity, DG CNECT), Head of CERT-EU, DORA Oversight Lead (ESAs).
* **Advisors:** Representatives from ENISA, NCIIPC, FS-ISAC.

---

### **Agenda: India-EU Financial Cybersecurity Preparatory Dialogue**
**Date:** November 4, 2025 | **Time:** 09:00-15:00 CET / 13:30-19:30 IST | **Format:** Hybrid

| Time (CET) | Session | Key Decision Point | Pre-Reads |
| :--- | :--- | :--- | :--- |
| **09:00-09:15** | **Opening Remarks & Agenda Adoption**<br/>*Co-Chairs* | Adopt the agenda and confirm the objective: finalize three core deliverables for the December 4 Dialogue. | 1. Draft Agenda<br/>2. Summary of 2025 India-EU Cyber Dialogues |
| **09:15-10:30** | **Workstream 1: Threat Intelligence Sharing Framework**<br/>*Owners: CERT-In, CERT-EU* | **Decision 1: Approve the draft MoU for the TI-Sharing Framework.**<br/>- Confirm legal basis (GDPR Art. 49, IT Act 70B).<br/>- Agree on TLP:RED channel for CERT-to-CERT sharing.<br/>- Endorse STIX/TAXII/MISP as the technical standard. | 3. Draft MoU for TI-Sharing Framework<br/>4. NIST SP 800-150 (Guide to CTI Sharing)<br/>5. FIRST TLP 2.0 Specification |
| **10:30-12:00** | **Workstream 2: Regulatory Convergence Roadmap**<br/>*Owners: RBI, ESAs* | **Decision 2: Endorse the 24-month Regulatory Convergence Roadmap.**<br/>- Agree on Q1 2026 milestone: Harmonized incident reporting template.<br/>- Agree on Q3 2026 milestone: Joint supervisory protocol for CTPPs.<br/>- Prioritize top 5 areas for convergence (reporting, testing, vendor risk). | 6. Draft Regulatory Convergence Roadmap (Gantt Chart)<br/>7. Side-by-Side Analysis: DORA vs. RBI/CERT-In Mandates<br/>8. DORA RTS/ITS Package (Incident Reporting, TLPT) |
| **12:00-13:00** | *Working Lunch* | | |
| **13:00-14:00** | **Workstream 3: Joint Initiatives & Pilots**<br/>*Owners: NCIIPC, ENISA* | **Decision 3: Lock the scope and funding model for three joint pilots.**<br/>- Pilot 1: AI-Assisted Cross-Border SOC.<br/>- Pilot 2: Zero Trust for Cross-Border Payments.<br/>- Pilot 3: DLT for Trade Finance & KYC.<br/>**Decision 4: Set targets for the "Cyber Fellowship 500" program.** | 9. Draft Pilot Charters (Objectives, Metrics, Timelines)<br/>10. Proposal for "Cyber Fellowship 500" Program<br/>11. EU AI Act (High-Risk Systems) & RBI AI Mandates |
| **14:00-14:45** | **Workstream 4: Joint Statement & Crisis Communications**<br/>*Owners: MEA, DG ECFIN* | **Decision 5: Adopt the draft Joint Statement and pre-agreed crisis communication scripts.**<br/>- Finalize language on shared threats and commitment to cooperation.<br/>- Approve "golden hour" holding statements for a major cross-border incident. | 12. Draft India-EU Joint Statement on Financial Cyber Resilience<br/>13. Draft Crisis Communication Playbook (Templates)<br/>14. FSB Guidance on Cyber Incident Communications |
| **14:45-15:00** | **Next Steps & Closing Remarks**<br/>*Co-Chairs* | Confirm intersessional milestones for finalizing texts by November 20. Assign action items for legal scrub and translation. | 15. Draft 90-Day Milestone Plan |

---

### Draft Deliverable Snippets for Discussion

**1. Joint Statement (Draft Snippet):**
> "Recognizing our shared vulnerability to sophisticated cyber threats targeting our interconnected financial systems, India and the European Union commit to a new era of strategic cooperation on financial cyber resilience. We will launch a **Regulatory Convergence Roadmap** to harmonize our respective legal frameworks, establish a **real-time Threat Intelligence Sharing Framework** between our CERTs, and co-sponsor **joint initiatives** in critical areas such as AI-driven security, Zero Trust Architecture, and secure Distributed Ledger Technologies. This partnership will be underpinned by the newly formed **India-EU BFSI Cyber Council**, a public-private body dedicated to our collective defense."

**2. Regulatory Convergence Roadmap (Sample Milestone):**
> **Milestone 1.1 (Q1 2026):** Harmonize Incident Reporting. **Owner:** RBI/ESAs Joint Working Group. **Deliverable:** A single, unified incident reporting template and a "report-once, distribute-many" protocol that satisfies both DORA's multi-stage timeline and CERT-In's 6-hour mandate.

**3. Threat Intelligence Sharing Framework (Draft MoU Snippet):**
> **Section 3: Information Handling.** All shared Cyber Threat Information (CTI) shall be marked according to the FIRST Traffic Light Protocol (TLP) v2.0. Information marked **TLP:RED** shall be exchanged exclusively between CERT-In and CERT-EU via a secure, dedicated TAXII 2.1 channel for the sole purpose of immediate threat mitigation and shall not be disseminated further without the explicit permission of the originating party.

## 11. Recommendations for the India-EU Dialogue

To translate strategic intent into tangible outcomes, the India-EU Macro Economic Dialogue on December 4, 2025, should focus on endorsing a concrete, time-bound action plan. The following recommendations provide a ready-made playbook for the preparatory meeting and the main dialogue, designed to produce a joint statement, a regulatory convergence roadmap, and a framework for threat intelligence sharing.

### A. Immediate Actions for the November 4th Preparatory Meeting

The preparatory meeting must be a decision-making forum, not a discussion group. The primary goal is to finalize draft deliverables for the main dialogue.

**1. Finalize and Approve the Threat Intelligence Sharing MoU:**
* **Action:** Table a pre-drafted Memorandum of Understanding (MoU) for a real-time threat intelligence sharing framework between CERT-In, CERT-EU, ENISA, and FS-ISAC.
* **Decision Point:** Secure approval on the MoU's core tenets: a **TLP:RED** channel for government-to-government sharing, mandatory use of **STIX/TAXII** standards, and a "privacy-by-design" data sanitization protocol.
* **Owner:** Joint leadership from CERT-In and CERT-EU.

**2. Endorse the Regulatory Convergence Roadmap:**
* **Action:** Present a 24-month Gantt chart outlining a phased roadmap for harmonizing key regulatory requirements between DORA and the RBI/CERT-In mandates.
* **Decision Point:** Agree on the top three priority workstreams for Year 1: **(1)** Incident Reporting Harmonization, **(2)** Joint TLPT/Red-Teaming Scenarios, and **(3)** a common framework for Third-Party/Cloud Vendor Risk Assessment.
* **Owner:** Joint leadership from RBI and the European Supervisory Authorities (ESAs).

**3. Lock the Scope for Three Joint Technology Pilots:**
* **Action:** Present three one-page charters for co-sponsored technology pilots.
* **Decision Point:** Approve the scope, success metrics, and funding models for pilots in: **(1)** AI-Assisted Cross-Border SOC, **(2)** Zero Trust for Cross-Border Payments, and **(3)** DLT for Trade Finance.
* **Owner:** Joint leadership from NCIIPC and ENISA.

**4. Adopt a Draft Joint Statement and Crisis Communication Playbook:**
* **Action:** Circulate a draft G7-style joint statement and a "break-glass" crisis communication playbook.
* **Decision Point:** Agree on the top-line commitments for the joint statement and approve pre-scripted "golden hour" holding statements for a major cross-border cyber incident to ensure coordinated public messaging.
* **Owner:** Joint leadership from India's Ministry of External Affairs (MEA) and the EU's DG ECFIN.

### B. Strategic Deliverables for the December 4th Dialogue

The main dialogue should formalize the agreements reached in the preparatory meeting.

**1. Sign the Joint Statement on Financial Cyber Resilience:**
* Publicly commit to the strategic partnership, highlighting the establishment of the Regulatory Convergence Roadmap, the TI-Sharing Framework, and the India-EU BFSI Cyber Council.

**2. Launch the India-EU BFSI Cyber Council:**
* Formally announce the creation of the Council, based on the CIISI-EU model, with a clear charter, governance structure, and a 90-day standing agenda for tactical, operational, and strategic collaboration.

**3. Publish the Regulatory Convergence Roadmap:**
* Release the 24-month roadmap with clear milestones, owners, and a commitment to a joint progress review every six months. The first milestone should be the delivery of a unified incident reporting template by the end of Q1 2026.

**4. Activate the Threat Intelligence Sharing Framework:**
* Formally sign the MoU and activate the secure STIX/TAXII channel between CERT-In and CERT-EU, initiating the real-time exchange of IoCs.

**5. Announce the "Cyber Fellowship 500" Program:**
* Launch the joint capacity-building initiative to address the critical skills gap, with a clear 12-month curriculum and defined funding streams from the Digital Europe Programme, Horizon Europe, and India's NSDC.

### C. Proposed Intersessional Milestones (Nov 4 - Dec 4, 2025)

To ensure readiness for the December 4 dialogue, a rapid, 30-day work plan is required.

| Milestone | Task | Owner | Deadline |
| :--- | :--- | :--- | :--- |
| **1** | Circulate final draft deliverables from Nov 4 meeting for legal scrub. | Co-Chairs' Offices | Nov 10, 2025 |
| **2** | Legal teams from India (MEA/MoL) and EU (Commission Legal Service) provide final input. | Legal Teams | Nov 20, 2025 |
| **3** | Finalize texts of Joint Statement, MoU, and Roadmap. | Co-Chairs' Offices | Nov 25, 2025 |
| **4** | Submit finalized documents for translation and inclusion in the Dec 4 Dialogue briefing book. | Co-Chairs' Offices | Nov 28, 2025 |
| **5** | Final pre-briefing for principals. | Co-Chairs | Dec 2, 2025 |

## 12. KPI & Dashboard Framework: Measuring Success

To ensure the India-EU cybersecurity cooperation is results-driven, a clear framework of Key Performance Indicators (KPIs) is essential. This framework should be tracked via a shared dashboard, with a regular reporting cadence to the India-EU BFSI Cyber Council and relevant regulators. The following 20 metrics provide a comprehensive system for measuring progress across the five core pillars of cooperation.

**Reporting Cadence:**
* **Monthly:** Tactical metrics (e.g., incident volumes, TI sharing speed) reviewed by the Council's operational working group.
* **Quarterly:** Strategic metrics (e.g., testing coverage, risk remediation, skills development) reviewed by the Council's leadership and reported to regulators (RBI, ESAs).

| Pillar | KPI | Metric Definition | Baseline (2025) | Target (2026) | Data Source / Owner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Incident Response** | **MTTD (Mean Time to Detect)** | Median time from threat actor initial access to internal detection. | 16 days (M-Trends) | < 24 hours | SOC Logs / CERTs |
| | **MTTR (Mean Time to Respond)** | Median time from detection to full incident containment and eradication. | 28 days (IBM) | < 72 hours | SOC Logs / CERTs |
| | **Reporting SLA Adherence (India)** | % of incidents reported to CERT-In within the 6-hour window. | 70% | > 98% | Compliance Logs / RBI |
| | **Reporting SLA Adherence (EU)** | % of incidents meeting DORA's 4h/24h/72h/1mo stages. | N/A (New) | > 98% | Compliance Logs / ESAs |
| **Threat Intelligence** | **TI Sharing Velocity** | Median time from IoC validation to sharing via the joint MISP/TAXII channel. | 48 hours | < 15 minutes | MISP Logs / Council Secretariat |
| | **TI Actionability Rate** | % of shared IoCs that are unique, relevant, and lead to a defensive action (e.g., blocklist update). | 25% | > 60% | SOC Logs / FS-ISAC |
| | **TI Contribution Ratio** | Ratio of members contributing intelligence vs. consuming it. | 1:10 | 1:3 | FS-ISAC / Council Secretariat |
| | **False Positive Rate** | % of shared alerts that are determined to be false positives after triage. | 15% | < 5% | SOC Logs / CERTs |
| **Testing Coverage** | **TLPT / Red Team Coverage** | % of systemically important FIs that have completed a joint or recognized TLPT/Red Team exercise. | 5% | 33% (per DORA's 3-yr cycle) | Council / Regulators |
| | **Vulnerability Remediation SLA** | % of "Critical" vulnerabilities (CVSS >9.0) patched within 14 days. | 55% | > 90% | VA/PT Reports / CISO Dashboards |
| | **Critical Asset Test Coverage** | % of designated "critical or important functions" covered by annual penetration testing. | 80% | 100% | Audit Reports / CISO Dashboards |
| | **DR Drill Success Rate** | % of Disaster Recovery drills that meet defined RTO/RPO targets. | 75% | > 95% | BCP/DR Reports / CIO Dashboards |
| **Supplier Risk** | **CTPP/Vendor Risk Remediation** | % of high-risk findings from CTPP/vendor audits remediated within 90 days. | 40% | > 85% | GRC Platforms / Regulators |
| | **SBOM Coverage** | % of critical applications with a complete and validated Software Bill of Materials (SBOM). | 10% | > 90% | DevSecOps Tools / CIO Dashboards |
| | **Exit Plan Test Rate** | % of critical cloud service contracts with a successfully tested exit plan. | 0% | 50% | BCP/DR Reports / CIO Dashboards |
| **Workforce & Skills** | **Critical Role Fill Rate** | % of open requisitions for critical roles (Threat Hunter, Cloud Sec, Red Team, IR) filled within 90 days. | 45% | > 80% | HR Metrics / CISO Dashboards |
| | **Fellowship Placement Rate** | % of "Cyber Fellowship 500" graduates placed in relevant BFSI roles. | N/A (New) | 95% | Council / HR |
| | **Board-Level Training** | % of Board members who have completed annual cybersecurity strategy and risk training. | 60% | 100% | Compliance Logs / Corporate Sec. |
| | **Certification Ratio** | % of cybersecurity staff holding at least one relevant professional certification (e.g., GCTI, OSCP, CCSP). | 35% | > 75% | HR Metrics / CISO Dashboards |
| | **Cyber Range Training Hours** | Total number of person-hours spent in hands-on cyber range training exercises per quarter. | 500 | > 5,000 | Training Logs / Council |

## 13. Risk Register: Navigating the Path to Cooperation

Implementing a comprehensive India-EU cybersecurity cooperation framework is not without its challenges. A proactive approach to identifying and mitigating potential risks is crucial for success. The following register outlines the top strategic risks to the initiative and proposes concrete mitigation strategies.

| Risk ID | Risk Description | Likelihood | Impact | Mitigation Strategy & Owner |
| :--- | :--- | :--- | :--- | :--- |
| **R-01** | **Regulatory Slippage:** Delays in finalizing the India-EU MoU or the Regulatory Convergence Roadmap due to political or bureaucratic hurdles, causing the initiative to lose momentum post-dialogue. | Medium | High | **Mitigation:** Secure high-level political endorsement in the Dec 4 Joint Statement. Establish a firm 90-day post-dialogue timeline for the BFSI Cyber Council to deliver its first set of harmonized templates. <br/>**Owner:** Co-Chairs (DEA/DG ECFIN). |
| **R-02** | **Data Privacy Conflicts:** Legal challenges to the threat intelligence sharing framework, arguing that it violates GDPR or India's DPDP Act, leading to a suspension of data flows. | Medium | High | **Mitigation:** Anchor the TI-Sharing MoU in GDPR Recital 49 (legitimate interest for security) and India's IT Act. Implement strict, automated PII scrubbing and data minimization protocols. Conduct a joint DPIA pre-launch. <br/>**Owner:** Joint Legal Working Group (MEA/EC Legal Service). |
| **R-03**| **Vendor Lock-In & CTPP Resistance:** Critical ICT Third-Party Providers (CTPPs), particularly large cloud providers, resist enhanced audit rights and portability requirements, slowing down the implementation of joint supervisory protocols. | High | High | **Mitigation:** Leverage the combined regulatory weight of the ESAs (under DORA) and the RBI. Use the EU Data Act's egress fee elimination as a lever. Prioritize pooled audits to reduce the burden on CTPPs. <br/>**Owner:** Joint Supervisory Working Group (RBI/ESAs). |
| **R-04** | **Skills Attrition:** High demand for talent leads to graduates of the "Cyber Fellowship 500" program being poached by non-BFSI sectors or other regions, failing to close the gap where it is most needed. | High | Medium | **Mitigation:** Implement a "service commitment" model where participating financial institutions sponsor fellows in exchange for a minimum 2-year employment term post-graduation. Offer competitive, benchmarked compensation packages. <br/>**Owner:** BFSI Cyber Council HR Sub-Committee. |
| **R-05** | **"Tick-Box" Compliance:** Financial institutions focus on meeting the letter of the harmonized regulations (e.g., reporting within 6 hours) without improving underlying security posture, leading to a false sense of security. | High | Medium | **Mitigation:** Link compliance to tangible outcomes in the KPI Dashboard. Make TLPT/Red Team exercise results (not just pass/fail) a key discussion point in quarterly reviews with regulators. <br/>**Owner:** Regulators (RBI/ESAs). |
| **R-06** | **Intelligence Overload:** The volume of shared threat intelligence overwhelms the analytical capacity of participating SOCs, leading to valuable alerts being missed (alert fatigue). | Medium | Medium | **Mitigation:** Invest in the AI-Assisted SOC pilot to automate triage and correlation. Use the Council to define clear "Priority Intelligence Requirements" (PIRs) to focus collection and analysis efforts. <br/>**Owner:** Council Operational Working Group (CERTs/FS-ISAC). |
| **R-07** | **Pilot Program Failure:** Joint technology pilots (AI-SOC, ZTA, DLT) fail to demonstrate clear ROI or encounter insurmountable technical/governance hurdles, undermining support for broader adoption. | Medium | Medium | **Mitigation:** Define clear, measurable success metrics from the outset. Adopt a "fail-fast" approach with 6-month check-ins to pivot or terminate underperforming pilots. Ensure strong project management from the Council Secretariat. <br/>**Owner:** Council Technology Sub-Committee (NCIIPC/ENISA). |
| **R-08** | **Funding Instability:** Bilateral funding for the Council and its initiatives (fellowships, pilots) is not sustained beyond the initial 12-18 months, causing programs to stall. | Medium | High | **Mitigation:** Develop a multi-year funding model that transitions from initial government seed funding to a sustainable model based on tiered membership dues from private sector participants, similar to FS-ISAC. <br/>**Owner:** Council Governance & Finance Sub-Committee. |

## 14. Communications & Legal-Hold Playbook: A "Golden Hour" Protocol

During a major cross-border cyber incident, the first 60 minutes—the "golden hour"—are critical for managing the narrative, containing market panic, and preserving legal integrity. A pre-agreed crisis communications and legal-hold playbook is therefore an essential component of the India-EU cooperation framework. This protocol must be designed to work in parallel with, but distinct from, the technical incident response.

### Guiding Principles

1. **Coordinated but Separate:** Indian and EU authorities will coordinate messaging but issue statements through their own channels to respect jurisdictional sovereignty.
2. **Transparency with Prudence:** Be as transparent as possible without compromising the ongoing investigation, creating undue panic, or revealing security vulnerabilities.
3. **Single Source of Truth:** Internally, all communications must flow through a designated Crisis Communications Lead on each side to ensure consistency.
4. **Preserve Everything:** A legal hold must be initiated immediately to prevent the inadvertent destruction of evidence crucial for regulatory investigations and potential litigation.

### Crisis Communication Workflow & Templates

| Time from Incident Declaration | Action | Owner | Template/Content Snippet |
| :--- | :--- | :--- | :--- |
| **T+0 to T+1 Hour (The Golden Hour)** | **Internal Activation & Initial Regulatory Notice:**<br/>- Activate the internal crisis communications team.<br/>- Issue immediate, confidential notification to lead regulators (RBI/ECB) and CERTs, triggering the EU-SCICF and India's CCMP. | CISO / Chief Comms Officer | **Regulator Notice (Email):**<br/>`Subject: URGENT/CONFIDENTIAL - [Financial Institution Name] - Potential Major Cyber Incident`<br/>`This is a confidential notice under [DORA Art. 19 / RBI Framework]. We are investigating a potential major cyber incident detected at [Time/Date]. Initial indicators suggest [e.g., ransomware, data exfiltration]. Containment is in progress. A formal report will follow within the [4/6]-hour window. PoC: [Name, Contact].` |
| **T+1 to T+4 Hours** | **Public Holding Statement & Media Response:**<br/>- Issue a pre-approved holding statement on official channels (website, social media).<br/>- Prepare a reactive Q&A for media inquiries. | Chief Comms Officer | **Public Holding Statement:**<br/>`[Financial Institution Name] has experienced a technology issue. We are working urgently to investigate and resolve the situation. Our top priority is the security of our clients' data and assets. We have alerted the relevant regulatory authorities and will provide further updates as soon as we have more information.` |
| **T+4 to T+24 Hours** | **Client & Investor Communication:**<br/>- Send targeted notifications to directly affected clients, if any.<br/>- Prepare a market disclosure if the incident is deemed "material" under SEBI LODR / EU MAR. | Head of Client Relations / Investor Relations | **Client Notification (Email/SMS):**<br/>`Important Security Update: We are investigating a cybersecurity incident. As a precaution, we recommend [e.g., monitoring your account, changing your password]. We are working to restore all services and will contact you directly if we identify any specific impact on your account.` |
| **T+24 to T+72 Hours** | **Detailed Updates & Regulatory Reporting:**<br/>- Provide more detailed, factual updates to the public and clients.<br/>- Submit the formal intermediate/72-hour reports to regulators. | Chief Comms Officer / CISO | **Public Update:**<br/>`Update on Technology Issue: Our investigation confirms we were the target of a sophisticated cyber incident. We have contained the threat and are working with leading cybersecurity experts to restore all services securely. We have found no evidence of [e.g., impact on client funds]. We continue to coordinate with regulatory authorities.` |

### Legal Hold & Evidence Preservation Protocol

This protocol must be initiated by the General Counsel immediately upon incident declaration.

**1. Legal Hold Notice:**
* **Action:** Issue a formal, written Legal Hold Notice to all relevant employees, contractors, and third-party service providers.
* **Template Snippet:**
 > `LEGAL HOLD - PRESERVE ALL DATA: You are hereby instructed to preserve all electronic and physical data related to [Incident Name/Date]. This includes, but is not limited to, all emails, chat logs, server logs, network traffic data, reports, notes, and drafts. Do not delete, alter, or destroy any potentially relevant information. This legal hold is effective immediately and remains in place until you are formally notified of its release.`

**2. Chain of Custody:**
* **Action:** The incident response team must maintain a strict chain of custody for all digital evidence collected.
* **Checklist:**
 * [ ] **Isolate:** Isolate affected systems to prevent tampering.
 * [ ] **Image:** Create bit-for-bit forensic images of all relevant drives and memory.
 * [ ] **Hash:** Document cryptographic hash values (e.g., SHA-256) for all original and copied evidence.
 * [ ] **Log:** Maintain a detailed log of who handled the evidence, when, and for what purpose.
 * [ ] **Secure:** Store all evidence in a secure, access-controlled location.
* **Compliance:** This process is critical for ensuring evidence is admissible under India's **Evidence Act, Section 65B** and meets the standards of the **Council of Europe's Budapest Convention** [CITE_NOT_FOUND][CITE_NOT_FOUND]. The log retention requirements of **CERT-In (180 days)** and **DORA** must be strictly followed [4] [CITE_NOT_FOUND].

## 15. Next 90-Day Milestones: From Dialogue to Delivery

The success of the December 4th Macro Economic Dialogue will be measured by the concrete actions that follow. A clear, ambitious 90-day plan is essential to maintain momentum and translate the high-level commitments into tangible progress. The following Gantt chart outlines the critical path from the preparatory meeting to the initial outputs of the newly formed India-EU BFSI Cyber Council.

**Objective:** To operationalize the three core deliverables—the Joint Statement, the Regulatory Convergence Roadmap, and the Threat Intelligence Sharing Framework—within 90 days of the December 4th summit.

---

### **India-EU Financial Cyber Resilience: 90-Day Implementation Plan (November 2025 - February 2026)**

| ID | Task | Owner | Nov '25 | Dec '25 | Jan '26 | Feb '26 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1.0** | **Pre-Dialogue Finalization** | | | | | |
| 1.1 | **Preparatory Meeting & Deliverable Approval** | Co-Chairs (DEA/ECFIN) | ████ | | | |
| 1.2 | Legal Scrub & Translation of Final Texts | Joint Legal Teams | ████████ | | | |
| 1.3 | **India-EU Macro Economic Dialogue & Summit Sign-off** | Principals | | ████ | | |
| **2.0** | **India-EU BFSI Cyber Council Launch** | | | | | |
| 2.1 | Formal Inauguration & Adoption of Charter | Council Co-Chairs | | ████ | | |
| 2.2 | Establish Secretariat & Working Groups (Ops, Tech, Policy) | Council Secretariat | | | ████████ | |
| 2.3 | First Quarterly Strategic Review Meeting | Council Leadership | | | | ████ |
| **3.0** | **Workstream 1: Threat Intelligence Sharing** | | | | | |
| 3.1 | Activate Secure MISP/TAXII Channel | CERT-In / CERT-EU | | ████████ | | |
| 3.2 | Begin Daily Exchange of Anonymized IoCs (TLP:AMBER) | Ops Working Group | | | ████ | |
| 3.3 | Conduct First Joint Threat Briefing | Ops Working Group | | | | ████ |
| **4.0** | **Workstream 2: Regulatory Convergence** | | | | | |
| 4.1 | Establish Joint Working Group (RBI/ESAs) | Policy Working Group | | | ████ | |
| 4.2 | **Deliverable: Harmonized Incident Reporting Template** | Policy Working Group | | | | ████████ |
| **5.0** | **Workstream 3: Joint Initiatives** | | | | | |
| 5.1 | Finalize Charters & Budgets for 3 Pilots | Tech Working Group | | | ████ | |
| 5.2 | Launch "Cyber Fellowship 500" Application Portal | HR Sub-Committee | | | | ████ |

---

**Key Milestones & Deliverables:**

* **Day 0 (Nov 4, 2025):** Drafts of all three core deliverables (Joint Statement, Roadmap, TI Framework) are approved at the preparatory meeting.
* **Day 30 (Dec 4, 2025):** Principals sign the Joint Statement at the Macro Economic Dialogue. The India-EU BFSI Cyber Council is formally launched.
* **Day 60 (approx. Jan 4, 2026):** The secure threat intelligence channel between CERT-In and CERT-EU is activated. The Council's working groups are established and hold their first meetings.
* **Day 90 (approx. Feb 4, 2026):** The Regulatory Convergence working group delivers the first key deliverable: a harmonized template for incident reporting that satisfies both DORA and CERT-In requirements. The "Cyber Fellowship 500" program is officially opened for applications.

## References

1. *Joint Technical Standards on major incident reporting*. https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/operational-resilience/joint-technical-standards-major-incident-reporting
2. *Preparing for DORA: ESAs Publish Incident Reporting ...*. https://www.morganlewis.com/blogs/sourcingatmorganlewis/2024/08/preparing-for-dora-esas-publish-incident-reporting-requirements
3. *Digital Operational Resilience Act (DORA) - EIOPA*. https://www.eiopa.europa.eu/digital-operational-resilience-act-dora_en
4. *Page 1 of 8 No. 20(3)/2022-CERT-In Government of India ...*. https://www.cert-in.org.in/PDF/CERT-In_Directions_70B_28.04.2022.pdf
5. *CERT-In Directions on Cybersecurity: An Explainer*. https://internetfreedom.in/cert-in-guidelines-on-cybersecurity-an-explainer/
6. *Cyber Security Outlook 2025*. https://www.dsci.in/files/content/documents/2025/Cyber-Security-Outlook-2025_0.pdf
7. *Ransomware attack on ION Group impacts derivatives ...*. https://www.bleepingcomputer.com/news/security/ransomware-attack-on-ion-group-impacts-derivatives-trading-market/
8. *ION brings clients back online after ransomware attack*. https://www.reuters.com/technology/ion-starts-bring-clients-back-online-after-ransomware-attack-source-2023-02-07/
9. *Ransomware attack on data firm ION could take days to fix -sources*. https://www.reuters.com/technology/ransomware-attack-data-firm-ion-could-take-days-fix-sources-2023-02-02/
10. *MOVEit breach: over 1000 organizations and 60 million ...*. https://www.itgovernanceusa.com/blog/moveit-breach-over-1000-organizations-and-60-million-individuals-affected
11. *Cybersecurity Skills Academy - Digital Skills and Jobs Platform*. https://digital-skills-jobs.europa.eu/en/cybersecurity-skills-academy
12. *Bridging India's cybersecurity skill gap - Mint*. https://www.livemint.com/mint-lounge/business-of-life/india-cybersecurity-skill-gap-11754807365228.html
13. *New CERT-In Guidelines 2025: What Every Security Team ...*. https://strobes.co/blog/new-cert-in-guidelines-2025-what-every-security-team-needs-to-act-on-now/
14. *Threat-Led Penetration Testing (TLPT) Under DORA Is ...*. https://treccert.com/threat-led-penetration-testing-tlpt-under-dora-is-now-in-effect/
15. *Data Breach Reporting in India: Legal obligations and Best ...*. https://ssrana.in/articles/data-breach-reporting-in-india-legal-obligations-and-best-practices/
16. *RFP for Implementation of Integrated Cash Management ...*. https://centralbankofindia.co.in/sites/default/files/2025-03/RFP_CMS_0.pdf
17. *RBI's Cybersecurity Mandates 2025: Securing India's Digital Banks*. https://www.jisasoftech.com/rbis-cybersecurity-mandates-2025-securing-indias-digital-banks/
18. *The Rise of AI-Powered Cyberattacks: Is BFSI Ready?*. https://www.cyberdefensemagazine.com/the-rise-of-ai-powered-cyberattacks-is-bfsi-ready/
19. *AI Act | Shaping Europe's digital future - European Union*. https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
20. *[PDF] ESMA75-117376770-460 Report on the functioning and review of ...*. https://www.esma.europa.eu/sites/default/files/2025-06/ESMA75-117376770-460_Report_on_the_functioning_and_review_of_the_DLTR_-_Art.14.pdf
21. *National Payment Corporation of India Launches Vajra, the ...*. https://www.blockchain-council.org/blockchain/national-payment-corporation-of-india-launches-vajra-the-blockchain-based-platform/
22. *Financial Services Information Sharing and Analysis Center (FS-ISAC)*. https://www.fsisac.com/
23. *Mission, structure and governing bodies*. https://www.certfin.it/about-us/
24. *Financial sector cyber collaboration centre (FSCCC - NCSC.GOV.UK*. https://www.ncsc.gov.uk/information/financial-sector-cyber-collaboration-centre-fsccc
25. *Euro Cyber Resilience Board for pan-European Financial ...*. https://www.ecb.europa.eu/paym/groups/euro-cyber-board/html/index.en.html
26. *[PDF] Regulation (EU) No 596/2014 of the European Parliament and of the ...*. https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32014R0596
27. *India Cybersecurity Domestic Market 2023 Report*. https://www.dsci.in/files/content/knowledge-centre/2023/India%20Cybersecurity%20Domestic%20Market%202023%20Report.pdf
28. *India BFSI Threat Landscape Report 2025 - SOCRadar*. https://socradar.io/resources/report/india-bfsi-threat-landscape-report-2025/
29. *India Cyber Threat Report 2025*. https://www.dsci.in/files/content/knowledge-centre/2024/India-Cyber-Threat-Report-2025.pdf
30. *ENISA THREAT LANDSCAPE 2025*. https://www.enisa.europa.eu/sites/default/files/2025-10/ENISA%20Threat%20Landscape%202025_0.pdf
31. *ENISA Threat landscape: Finance sector*. https://www.enisa.europa.eu/sites/default/files/2025-02/Finance%20TL%202024_Final.pdf
32. *Digital Threat Report 2024*. https://www.cert-in.org.in/PDF/Digital_Threat_Report_2024.pdf
33. *Significant Cyber Incidents | Strategic Technologies Program*. https://www.csis.org/programs/strategic-technologies-program/significant-cyber-incidents
34. *L'ANSSI publie le Panorama de la cybermenace 2023*. https://cyber.gouv.fr/actualites/lanssi-publie-le-panorama-de-la-cybermenace-2023
35. *KillNet Showcases New Capabilities While Repeating ...*. https://cloud.google.com/blog/topics/threat-intelligence/killnet-new-capabilities-older-tactics
36. *India BFSI Industry Threat Landscape Report*. https://socradar.io/wp-content/uploads/2025/08/India-BFSI-Report.pdf
37. *Global Cybersecurity Outlook 2025*. https://reports.weforum.org/docs/WEF_Global_Cybersecurity_Outlook_2025.pdf
38. *Hackers who breached ION say ransom paid; company declines ...*. https://www.reuters.com/technology/hackers-say-ransom-paid-case-derivatives-data-firm-ion-company-declines-comment-2023-02-03/
39. *Snowflake, Ticketmaster & Santander Breaches: A Live ...*. https://www.cm-alliance.com/cybersecurity-blog/snowflake-ticketmaster-santander-breaches-a-live-timeline
40. *More than 12000 Santander employees in US affected by ...*. https://therecord.media/santander-employees-bank-breach-affected
41. *Capita fined £14m for data breach affecting over 6m people*. https://ico.org.uk/about-the-ico/media-centre/news-and-blogs/2025/10/capita-fined-14m-for-data-breach-affecting-over-6m-people/
42. *Capita Cyber Security Breach – £14 Million Fine Issued | Insights*. https://www.mayerbrown.com/en/insights/publications/2025/10/capita-cyber-security-breach-14-million-pounds-fine-issued
43. *European Investment Bank (EIB) suffers cyberattack*. https://www.incibe.es/en/incibe-cert/publications/cybersecurity-highlights/european-investment-bank-eib-suffers-cyberattack
44. *Regulation - 2022/2554 - EN - DORA - EUR-Lex*. https://eur-lex.europa.eu/eli/reg/2022/2554/oj/eng
45. *NIS2 vs. DORA: Key differences & why they matter - Diligent*. https://www.diligent.com/resources/blog/nis2-vs-dora
46. *L_2022333EN.01000101.xml - EUR-Lex - European Union*. https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32022R2554
47. *DORA regulation | Digital operational resilience in the financial sector*. https://www.springlex.eu/en/packages/dora/dora-regulation/
48. *Digital operational resilience for the financial sector - EUR-Lex*. https://eur-lex.europa.eu/EN/legal-content/summary/digital-operational-resilience-for-the-financial-sector.html
49. *DORA – New Digital Operational Resilience rules for the EU's ...*. https://www.hoganlovells.com/en/publications/dora-new-digital-operational-resilience-rules-for-the-eus-financial-sector
50. *RBI Master Direction on Information Technology ...*. https://seconize.co/blog/rbi-master-direction-on-information-technology/
51. *[PDF] RESERVE BANK OF INDIA - FIDC*. https://fidcindia.org.in/wp-content/uploads/2023/04/RBI-OUTSOURCING-OF-IT-SERVICES-10-04-23.pdf
52. *Summary: RBI guidelines on IT governance and cybersecurity*. https://www.medianama.com/2023/11/223-summary-rbi-direction-it-governance-risk-controls/
53. *Regulation (EU) 2022/2554 of the European Parliament and*. https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32022R2554
54. *RBI Compliance and the RBI Cyber Security Framework*. https://www.endpointprotector.com/blog/rbi-compliance-and-the-rbi-cyber-security-framework/
55. *DORA Contract Compliance: Essential Guide for Financial Firms*. https://www.3rdrisk.com/blog/dora-compliant-contracts
56. *RBI Pushes Zero Trust for Banking Cybersecurity*. https://www.seqrite.com/blog/rbi-emphasizes-adopting-zero-trust-approaches-for-banking-institutions/
57. *Data protection laws in India*. https://www.dlapiperdataprotection.com/index.html?t=law&c=IN
58. *Data protection adequacy for non-EU countries*. https://commission.europa.eu/law/law-topic/data-protection/international-dimension-data-protection/adequacy-decisions_en
59. *Recital 49 - Network and Information Security as Overriding ...*. https://gdpr-info.eu/recitals/no-49/
60. *Impact of the Digital Personal Data Protection (DPDP) Act ...*. https://www.dpo-india.com/Blogs/impact-dpdpa-cross-border/
61. *Guidance on Cross-Border Data Transfers for Indian ...*. https://www.dataguidance.com/sites/default/files/dcsi_privacy_across_borders-_guidance_on_cross-border_data_transfers_for_indian_organizations.pdf
62. *Transfer of personal data in India - Data Protection Laws of the World*. https://www.dlapiperdataprotection.com/?t=transfer&c=IN
63. *Collaboration - CERT-In*. https://www.cert-in.org.in/s2cMainServlet?pageid=Collaboration
64. *CSIRTs Network | ENISA - European Union*. https://www.enisa.europa.eu/topics/eu-incident-response-and-cyber-crisis-management/csirts-network
65. *STIX™ Version 2.1 - Index of / - OASIS Open*. https://docs.oasis-open.org/cti/stix/v2.1/csprd01/stix-v2.1-csprd01.html
66. *TAXII Version 2.1 - Index of / - OASIS Open*. https://docs.oasis-open.org/cti/taxii/v2.1/os/taxii-v2.1-os.html
67. *MISP Open Source Threat Intelligence Platform & Open Standards ...*. https://www.misp-project.org/
68. *Traffic Light Protocol (TLP)*. https://www.first.org/tlp/
69. *Operating Rules*. https://www.fsisac.com/operating_rules
70. *Cooperation with CERT-EU | ENISA - European Union*. https://www.enisa.europa.eu/topics/eu-cyber-crisis-and-incident-management/cooperation-with-cert-eu
71. *Extended Detection And Response Platforms, Q2 2024*. https://www.forrester.com/blogs/announcing-the-forrester-wave-extended-detection-and-response-platforms-q2-2024/
72. *Forrester Wave: Security Analytics Platforms, 2025: SIEM Vs XDR*. https://www.forrester.com/blogs/announcing-the-forrester-wave-security-analytics-platforms-2025-the-siem-vs-xdr-fight-intensifies/
73. *SP 800-207A, A Zero Trust Architecture Model for Access ...*. https://csrc.nist.gov/pubs/sp/800/207/a/final
74. *Top Use Cases in Preemptive Cyber Defense*. https://veriti.ai/blog/veriti-mentioned-in-the-2024-gartner-emerging-tech-top-use-cases-in-preemptive-cyber-defense/
75. *Transforming Cyber Deception From Optional to Essential*. https://www.gartner.com/en/documents/6663934
76. *Cyber-Defense-Magazine-May-2023.pdf*. https://www.netsfere.com/assets/Cyber-Defense-Magazine-May-2023.pdf
77. *What Is MITRE D3FEND?*. https://www.splunk.com/en_us/blog/learn/mitre-defend.html
78. *Regulation - 2022/858 - EN - dlt - EUR-Lex - European Union*. https://eur-lex.europa.eu/eli/reg/2022/858/oj/eng
79. *RBI's CBDC Retail Pilot Surpasses 60 Lakh Users, Introduces ...*. https://bfsi.economictimes.indiatimes.com/articles/rbis-cbdc-retail-pilot-surpasses-60-lakh-users-introduces-offline-and-programmable-features/121482944
80. *[PDF] JC 2025 29 Guide on DORA oversight activities*. https://www.esma.europa.eu/sites/default/files/2025-07/JC_2025_29__DORA_Guide_on_oversight_activities.pdf
81. *Digital Operational Resilience Act (DORA), Article 30*. https://www.digital-operational-resilience-act.com/Article_30.html
82. *Data Act explained | Shaping Europe's digital future*. https://digital-strategy.ec.europa.eu/en/factpages/data-act-explained
83. *भारतीय  रज़वर् ब RESERVE BANK OF INDIA*. https://fidcindia.org.in/wp-content/uploads/2023/11/RBI-IT-MASTER-DIRECTIONS-07-11-23.pdf
84. *Digital Operational Resilience Act (DORA), Article 28*. https://www.digital-operational-resilience-act.com/Article_28.html
85. *European Cybersecurity Skills Framework (ECSF) - ENISA*. https://www.enisa.europa.eu/topics/skills-and-competences/skills-development/european-cybersecurity-skills-framework-ecsf
86. *NICE Workforce Framework for Cybersecurity (NICE Framework)*. https://niccs.cisa.gov/tools/nice-framework
87. *European Cybersecurity Skills Framework Role Profiles.pdf*. https://www.enisa.europa.eu/sites/default/files/publications/European%20Cybersecurity%20Skills%20Framework%20Role%20Profiles.pdf
88. *Mind the Cyber Skills Gap: a deep-dive*. https://digital-skills-jobs.europa.eu/en/latest/briefs/mind-cyber-skills-gap-deep-dive
89. *ENISA Mandate and Regulatory Framework*. https://www.enisa.europa.eu/about-enisa/regulatory-framework/legislation
90. *Training and capacity building - Europol - European Union*. https://www.europol.europa.eu/how-we-work/services-support/training-and-capacity-building
91. *No. 20(3)/2022-CERT-In*. https://npstrust.org.in/sites/default/files/circulars-documents/Cyber_Security_Directions_and_FAQs_issued_by_CERT_In.pdf
92. *Membership | Learn More About FS-ISAC*. https://www.fsisac.com/membership
93. *CIISI-EU Community Rulebook - European Central Bank*. https://www.ecb.europa.eu/paym/groups/euro-cyber-board/shared/pdf/ciisi-eu_community_rulebook.pdf
94. *[PDF] CIISI-EU Terms of Reference - European Central Bank*. https://www.ecb.europa.eu/paym/groups/euro-cyber-board/shared/pdf/ciisi-eu_terms_of_reference.pdf