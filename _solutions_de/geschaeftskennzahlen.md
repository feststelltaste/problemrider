---
title: Geschäftskennzahlen
description: Definition von Geschäftskennzahlen zur Bewertung von Funktionalität
  und Qualität der Software.
category:
- Business
- Management
problems:
- declining-business-metrics
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- quality-blind-spots
- invisible-nature-of-technical-debt
- stakeholder-confidence-loss
- negative-brand-perception
- resource-waste
layout: solution
lang: de
en_slug: business-metrics
related_solutions:
- slug: code-metrics
  similarity: 0.8
- slug: total-cost-of-ownership-transparency
  similarity: 0.75
- slug: service-level-objectives
  similarity: 0.75
- slug: security-relevant-metrics
  similarity: 0.75
- slug: service-level-agreements
  similarity: 0.75
- slug: security-metrics
  similarity: 0.7
---

## Description

Geschäftskennzahlen sind messbare Indikatoren — Konversionsrate, Auftragsabwicklungszeit, Umsatz pro Sitzung — spezifisch definiert, um die Geschäftsergebnisse zu erfassen, die ein System unterstützen soll, direkt in das System instrumentiert, sodass sein tatsächliches Verhalten, nicht nur seine technischen Eigenschaften, über die Zeit beobachtet und verfolgt werden kann. Der Mechanismus erfordert enge Zusammenarbeit zwischen Geschäfts- und technischen Stakeholdern, um zu identifizieren, welche Ergebnisse tatsächlich wichtig sind, und dann oft leichtgewichtige Instrumentierung zu einem System hinzuzufügen, das möglicherweise nie darauf ausgelegt war, diese Art von Daten zu exponieren. Dies ist wichtig für Legacy-Modernisierung, weil die geschäftliche Auswirkung der Unzulänglichkeiten eines Legacy-Systems — langsame Seiten, gescheiterte Checkouts, manuelle Workarounds — üblicherweise lange qualitativ gefühlt wird, bevor sie jemand quantitativ formulieren kann, was das Team unfähig macht, Modernisierungsinvestition in Begriffen zu rechtfertigen, auf die Entscheidungsträger reagieren können, da technische Schulden und Systemverfall sonst in normaler Geschäftsberichterstattung unsichtbar sind. Eine Baseline vor Beginn der Modernisierungsarbeit zu etablieren und dieselben Kennzahlen anschließend zu verfolgen, verwandelt den Wert dieser Arbeit von einer angenommenen Verbesserung in eine demonstrierte. Das Risiko ist, dass schlecht gewählte Kennzahlen die falschen Optimierungen anreizen können, und dass die Definition genuin bedeutsamer Kennzahlen echte gemeinsame Anstrengung erfordert, statt einfach anzuschließen, was auch immer aus dem Legacy-System leicht zu extrahieren ist.

## How to Apply ◆

- Identifizieren Sie Schlüssel-Geschäftsergebnisse, die das Legacy-System unterstützt (Umsatzverarbeitung, Kunden-Onboarding-Zeit, Auftragsabwicklungsrate), und definieren Sie messbare Kennzahlen für jedes.
- Instrumentieren Sie das Legacy-System, um diese Kennzahlen zu sammeln, selbst wenn es das Hinzufügen leichtgewichtigen Monitoring-Codes erfordert.
- Etablieren Sie Baselines für aktuelle Kennzahlwerte, bevor Sie irgendeine Modernisierungsbemühung beginnen.
- Erstellen Sie Dashboards, die Geschäftskennzahlen sowohl für technische als auch für Geschäfts-Stakeholder sichtbar machen.
- Nutzen Sie Geschäftskennzahlen, um Modernisierungsarbeit zu priorisieren: Fokussieren Sie sich auf Bereiche, wo schlechte Systemqualität direkt Geschäftsergebnisse beeinflusst.
- Verfolgen Sie Kennzahlen über die Zeit, um den Wert von Modernisierungsinvestitionen zu demonstrieren.

## Tradeoffs ⇄

**Vorteile:**
- Bietet objektive Evidenz für Investitionsentscheidungen bei Legacy-Systemverbesserung.
- Richtet technische Arbeit an Geschäftswert aus, was es einfacher macht, Stakeholder-Unterstützung zu sichern.
- Offenbart die tatsächliche geschäftliche Auswirkung technischer Schulden und Legacy-Systembeschränkungen.
- Ermöglicht datengetriebene Priorisierung von Modernisierungsbemühungen.

**Kosten:**
- Die Definition bedeutsamer Kennzahlen erfordert enge Zusammenarbeit zwischen Geschäfts- und technischen Teams.
- Die Instrumentierung von Legacy-Systemen zur Kennzahlensammlung kann technisch herausfordernd sein.
- Schlecht gewählte Kennzahlen können die falschen Verhaltensweisen oder Optimierungen anreizen.
- Kennzahlensammlung fügt dem System Overhead hinzu, wenn auch typischerweise minimal.

## How It Could Be

Eine Legacy-E-Commerce-Plattform leidet unter langsamen Seitenladezeiten und häufigen Checkout-Fehlern, aber das Entwicklungsteam kämpft damit, Modernisierungsinvestition zu rechtfertigen, weil es die Auswirkung nicht quantifizieren kann. Sie definieren Geschäftskennzahlen: Konversionsrate, Warenkorbabbruchrate, durchschnittliche Seitenladezeit und Umsatz pro Sitzung. Nach der Instrumentierung des Legacy-Systems entdecken sie, dass Checkout-Fehler das Geschäft monatlich erheblichen Umsatz kosten und dass langsame Produktseitenladezeiten mit höheren Absprungraten korrelieren. Ausgestattet mit diesen Zahlen sichert das Team Finanzierung für gezielte Performance-Verbesserungen und kann nach jedem Modernisierungs-Sprint messbare Geschäftsverbesserung demonstrieren.
