---
title: Site Reliability Engineering (SRE)
description: Anwendung von Prinzipien für stabilen Systembetrieb.
category:
- Operations
- Process
problems:
- system-outages
- constant-firefighting
- slow-incident-resolution
- monitoring-gaps
- deployment-risk
- operational-overhead
- poor-operational-concept
- high-maintenance-costs
- cascade-failures
- developer-frustration-and-burnout
layout: solution
lang: de
en_slug: site-reliability-engineering-sre
related_solutions:
- slug: chaos-engineering
  similarity: 0.8
- slug: incident-management
  similarity: 0.8
- slug: service-level-objectives
  similarity: 0.8
- slug: error-budgets
  similarity: 0.8
- slug: secure-software
  similarity: 0.75
- slug: runbooks
  similarity: 0.75
---

## Description

Site Reliability Engineering ist eine Disziplin, die Softwareentwicklungs-Rigorosität auf den Betrieb anwendet und Zuverlässigkeit als eine Eigenschaft behandelt, die gemessen, budgetiert und systematisch verbessert werden kann, statt eine Frage manuellen Aufwands und Glücks zu sein. Ihre Kernmechanismen — Fehlerbudgets, die an Service Level Objectives gebunden sind, schuldlose Postmortems, eine harte Obergrenze für den Anteil der auf manuelles "Toil" verwendeten Zeit und progressive Rollout-Strategien — verwandeln zusammen den Betrieb von einer reaktiven, personenabhängigen Aktivität in eine Ingenieurspraxis mit Feedback-Schleifen. Diese Neurahmung adressiert ein Muster, das besonders häufig um Legacy-Systeme herum auftritt: betriebliches Wissen, das in einer oder zwei Personen konzentriert ist, Vorfälle, die durch Ad-hoc-Feuerwehrlöschen statt dokumentierte Prozeduren behandelt werden, und keine prinzipielle Möglichkeit zu entscheiden, wann Zuverlässigkeitsarbeit Priorität vor neuen Features haben sollte. Der Fehlerbudget-Mechanismus von SRE liefert genau diese fehlende Entscheidungsregel, während seine Toil-Automatisierungs- und Runbook-Praktiken das Wissen verteilen, das sonst im Kopf eines einzelnen Experten eingeschlossen bliebe. Die Übernahme von SRE für einen Legacy-Bestand ist ebenso sehr eine organisatorische wie eine technische Änderung, da sie erfordert, dass das Management akzeptiert, dass ein aufgebrauchtes Fehlerbudget echt Feature-Arbeit stoppt, und sie beginnt typischerweise damit, das Legacy-System gut genug zu instrumentieren, um die SLOs zu messen, von denen die gesamte Praxis abhängt.

## How to Apply ◆

> Legacy-Systemen fehlt häufig betriebliche Disziplin, was zu chronischem Feuerwehrlöschen und unvorhersehbarer Zuverlässigkeit führt. SRE-Prinzipien bringen Ingenieurs-Rigorosität in den Betrieb und behandeln betriebliche Arbeit als ein Softwareproblem, das gemessen, automatisiert und systematisch verbessert werden kann.

- Etablieren Sie Fehlerbudgets, die an Service Level Objectives für jeden kritischen Dienst gebunden sind. Wenn das Fehlerbudget aufgebraucht ist, verschieben Sie Entwicklungsaufwand von Feature-Entwicklung zu Zuverlässigkeitsarbeit. Dies schafft einen selbstregulierenden Mechanismus, der verhindert, dass Zuverlässigkeit dauerhaft deprioritisiert wird.
- Implementieren Sie einen schuldlosen Postmortem-Prozess für alle signifikanten Vorfälle. Dokumentieren Sie den Zeitverlauf, die Grundursache, die beitragenden Faktoren und konkrete Handlungspunkte. Fokussieren Sie auf systemische Verbesserungen statt individuelle Schuldzuweisung, um ehrliche Berichterstattung und Lernen zu fördern.
- Automatisieren Sie Toil — repetitive, manuelle betriebliche Aufgaben, die linear mit der Systemgröße skalieren. In Legacy-Systemen bedeutet dies oft die Automatisierung von Deployment-Prozeduren, Log-Analyse, Kapazitätsprüfungen und routinemäßigen Wartungsaufgaben, die Betreiberzeit verbrauchen, ohne dauerhaften Wert hinzuzufügen.
- Führen Sie Bereitschaftsdienst-Rotationen mit klaren Eskalationspfaden und Runbooks für bekannte Fehlermodi ein. Legacy-Systeme verlassen sich oft auf einen einzelnen Experten, der alle Vorfälle bearbeitet; die Verteilung dieser Verantwortung reduziert Single Points of Failure im Team.
- Wenden Sie das Prinzip der Reduzierung der mittleren Wiederherstellungszeit (MTTR) an, statt Null-Fehler anzustreben. Für Legacy-Systeme, die nicht leicht neu gestaltet werden können, sind schnelle Erkennung und Wiederherstellung erreichbarer und wirkungsvoller als die Verhinderung aller Fehler.
- Messen und begrenzen Sie den Prozentsatz der Entwicklungszeit, die für betrieblichen Toil aufgewendet wird (SRE empfiehlt eine Obergrenze von 50 %). Wenn betriebliche Arbeit diese Schwelle überschreitet, signalisiert dies, dass das System strukturelle Verbesserungen braucht, nicht mehr Feuerwehrleute.
- Implementieren Sie progressive Rollout-Strategien (Canary-Deployments, Feature-Flags), um den Blast-Radius von Änderungen in Systemen zu reduzieren, in denen Änderungen inhärent riskant sind.

## Tradeoffs ⇄

> SRE-Praktiken verwandeln den Betrieb von einem reaktiven Kostenzentrum in eine proaktive Ingenieursdisziplin, erfordern aber organisatorisches Engagement und kulturellen Wandel.

**Vorteile:**

- Reduziert chronisches Feuerwehrlöschen durch die Etablierung klarer Richtlinien dafür, wann Zuverlässigkeitsarbeit Priorität vor Feature-Entwicklung hat.
- Verteilt betriebliches Wissen über Teams hinweg durch Runbooks und Bereitschaftsdienst-Rotationen, was die Abhängigkeit von einzelnen Experten reduziert.
- Bietet messbare Kriterien für die Systemgesundheit durch SLOs und Fehlerbudgets, was Zuverlässigkeitsdiskussionen objektiv statt politisch macht.
- Beseitigt systematisch Toil durch Automatisierung und setzt im Laufe der Zeit Entwicklungskapazität für höherwertige Arbeit frei.
- Verbessert die Vorfallreaktion durch schuldlose Postmortems und gemeinsames Lernen, was wiederkehrende Vorfälle reduziert.

**Kosten und Risiken:**

- Erfordert erheblichen kulturellen Wandel, besonders in Organisationen, in denen Betrieb und Entwicklung getrennte Funktionen mit unterschiedlichen Anreizen sind.
- Fehlerbudget-Richtlinien können Reibung erzeugen, wenn Feature-Fristen mit Zuverlässigkeitsprioritäten kollidieren, was starke Managementunterstützung erfordert.
- Bereitschaftsdienst-Rotationen für Legacy-Systeme können belastend sein, wenn das System viele Fehlermodi und wenige Runbooks hat, was zu Burnout während der Übergangsphase führt.
- Die Automatisierung von Legacy-Betriebsaufgaben könnte erhebliche Investition in Tooling und Infrastruktur erfordern, die das Legacy-System nicht zu unterstützen konzipiert wurde.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie SRE-Prinzipien den Betrieb von Legacy-Systemen stabilisieren.

Ein Logistikunternehmen betreibt ein Legacy-Sendungsverfolgungssystem, das häufige Ausfälle erlebt, die jeweils denselben leitenden Ingenieur zum Diagnostizieren und Beheben erfordern. Das Unternehmen führt SRE-Praktiken ein, indem es zunächst die Top-10-Fehlermodi in Runbooks dokumentiert, dann eine Bereitschaftsdienst-Rotation unter vier Ingenieuren etabliert. Sie definieren ein SLO von 99,9 % Verfügbarkeit für die Tracking-API und messen es wöchentlich. Innerhalb von drei Monaten ermöglichen die Runbooks Nachwuchsingenieuren, 80 % der Vorfälle ohne Eskalation zu lösen, die Bereitschaftsdienstlast des leitenden Ingenieurs sinkt von jeder Nacht auf eine Woche von vier, und das Team identifiziert zwei systemische Probleme, die für 60 % aller Vorfälle verantwortlich sind. Die Behebung dieser zwei Probleme mit gezielter Automatisierung reduziert die monatliche Vorfallanzahl von 15 auf 4.

Eine Gesundheitsplattform betreibt ein Legacy-Patientenaktensystem, bei dem Deployments manuell von einem einzelnen Betriebsingenieur über ein Wochenende durchgeführt werden, was zu ausgedehnter Ausfallzeit und häufigen, Rollback erfordernden Fehlern führt. Das SRE-Team automatisiert die Deployment-Pipeline, führt Canary-Deployments ein, die Änderungen zuerst an 5 % des Traffics ausrollen, und etabliert automatisierte Health Checks, die einen automatischen Rollback auslösen, wenn Fehlerraten die Baseline um mehr als 2 % überschreiten. Die Deployment-Frequenz steigt von monatlich auf wöchentlich, Deployment-bezogene Vorfälle sinken um 90 %, und der Betriebsingenieur richtet seine Wochenendzeit auf den Bau von Monitoring-Infrastruktur um, die die Systemzuverlässigkeit weiter verbessert.
