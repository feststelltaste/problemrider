---
title: Definition of Done
description: Definition klarer Kriterien für die Fertigstellung von Funktionalität.
category:
- Process
- Testing
- Code
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/definition-of-done/
problems:
- poor-test-coverage
- insufficient-testing
- high-bug-introduction-rate
- quality-degradation
- inconsistent-quality
- high-defect-rate-in-production
- quality-compromises
- quality-blind-spots
- partial-bug-fixes
- lower-code-quality
- reduced-feature-quality
- inadequate-error-handling
- poor-documentation
- feature-creep-without-refactoring
- inadequate-initial-reviews
- inconsistent-execution
- increased-technical-shortcuts
- outdated-tests
- perfectionist-culture
- perfectionist-review-culture
- rushed-approvals
- bikeshedding
- change-management-chaos
- gold-plating
- no-formal-change-control-process
- feature-creep
- rapid-prototyping-becoming-production
layout: solution
lang: de
en_slug: definition-of-done
related_solutions:
- slug: definition-of-ready
  similarity: 0.75
- slug: code-quality-gates
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: code-review-process-reform
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
- slug: regression-testing
  similarity: 0.7
---

## Description

Eine Definition of Done ist ein expliziter, universeller Qualitätsmaßstab — Code überprüft, Tests bestanden, in eine Testumgebung deployt —, den jeder Arbeitsposten erfüllen muss, bevor er als fertig zählt, und ersetzt den impliziten, oft inkonsistenten Standard von „der Entwickler sagt, es funktioniert". In Legacy-Kontexten, wo „fertig" historisch genau das bedeutete, ist eine DoD häufig der erste Mechanismus, der halb fertige Änderungen davon abhält, still ein bereits fragiles System zu untergraben, und sie dient gleichzeitig als forcierender Faktor, der Lücken in der eigenen Qualitätsinfrastruktur des Teams aufdeckt — fehlende Testumgebungen, fehlende Rollback-Verifikation —, die ohne Adressierung die DoD nicht erfüllt werden kann. Die DoD auf das zu kalibrieren, was das Team jetzt realistisch erreichen kann, und sie bewusst zu erhöhen, während sich die Kompetenz verbessert, hält sie zu einem echten Standard statt zu einem Dokument, das unter Druck aufgehoben wird.

## How to Apply ◆

> In Legacy-Kontexten, wo „fertig" historisch „der Entwickler sagt, es funktioniert" bedeutete, ist eine formelle Definition of Done oft der erste Mechanismus, der teilweise fertige Änderungen davon abhält, still ein bereits fragiles System zu untergraben.

- Beginnen Sie mit einer minimalen DoD, die das Team bei jedem Arbeitsposten realistisch erfüllen kann, selbst unter Legacy-Druck: Code überprüft, alle bestehenden Tests bestehen weiterhin, Änderung in eine Testumgebung deployt. Fügen Sie Kriterien hinzu, während die Infrastruktur des Teams reift.
- Beziehen Sie explizit ein Kriterium „keine neuen Schulden ohne entsprechenden Backlog-Eintrag eingeführt" ein — dies macht die Anhäufung von Workarounds sichtbar, statt sie unter Zeitdruck durchschlüpfen zu lassen.
- Fügen Sie ein Regressionsprüfungskriterium hinzu, das verlangt, dass die Änderung gegen die spezifischen Legacy-Verhaltensweisen verifiziert wurde, die am wahrscheinlichsten gestört werden — Integrationspunkte mit externen Systemen, Batch-Jobs und geplante Prozesse sind häufige blinde Flecken.
- Verlangen Sie, dass jede Änderung, die einen undokumentierten Bereich berührt, mindestens einen kurzen Inline-Kommentar oder eine Entscheidungsaufzeichnung enthält, die erklärt, was der Code tut und warum — dies erzeugt inkrementelle Dokumentation, ohne einen separaten Dokumentations-Sprint zu erfordern.
- Beziehen Sie ein Rollback-Verifikationskriterium für Datenbank-berührende Änderungen ein: Die Migration muss rückwärts getestet werden, bevor die Story als fertig gilt, angesichts dessen, dass Legacy-Systeme selten automatisierte Rollback-Abdeckung haben.
- Unterscheiden Sie klar zwischen der DoD (dem universellen Qualitätsmaßstab, der auf jeden Arbeitsposten angewandt wird) und story-spezifischen Abnahmekriterien, die definieren, was jedes bestimmte Feature tun muss — Legacy-Teams vermengen dies oft und enden ohne dass eines von beidem ordentlich durchgesetzt wird.
- Überprüfen und erweitern Sie die DoD bei Retrospektiven, wann immer ein Produktionsvorfall auf etwas zurückgeführt werden kann, das „fertig" war, aber eine Qualitätsprüfung vermissen ließ — jeder Vorfall ist ein Beleg für eine Lücke in der DoD.
- Machen Sie die DoD im täglichen Arbeitsablauf des Teams sichtbar (Sprint-Board, Wiki, Team-Kanal), statt sie zu einem einmal geschriebenen und vergessenen Dokument zu machen, da Legacy-Teams unter ständigem Feuerlöschdruck nicht danach suchen werden, es sei denn, sie ist unvermeidlich.

## Tradeoffs ⇄

> Eine Definition of Done erzeugt kurzfristige Reibung im Austausch gegen den langfristigen Nutzen, zu verhindern, dass sich Qualität mit jedem Release schrittweise verschlechtert.

**Vorteile:**

- Verhindert das in Legacy-Projekten häufige „Härtungs-Sprint"-Muster, bei dem sich Monate aufgeschobener Qualitätsarbeit kurz vor einem Release anhäufen und Verzögerungen oder Qualitätskompromisse verursachen.
- Schafft eine gemeinsame Qualitätssprache über Entwickler, Tester und Betriebspersonal hinweg, die sonst möglicherweise grundlegend unterschiedliche implizite Definitionen von „fertig" haben, eine Lücke, die in langlebigen Legacy-Teams besonders groß ist.
- Macht unerledigte Arbeit sichtbar: Wenn das Team konsequent die DoD innerhalb eines Sprints nicht erfüllen kann, ist das ein Signal, dass der Umfang zu groß oder die Qualitätsinfrastruktur zu schwach ist, beides muss explizit adressiert werden.
- Erzwingt schrittweise Verbesserung der Qualitätsinfrastruktur des Legacy-Systems — Testumgebungen, Deployment-Pipelines, Dokumentationspraktiken —, weil die DoD Bedarf dafür schafft.
- Unterstützt schrittweise Erhöhungen des Qualitätsmaßstabs: Ein Team, das konsequent eine einfache DoD erfüllt, kann sie anheben, was einen Sperrklinkenmechanismus für kontinuierliche Verbesserung schafft statt periodischer Bereinigungskampagnen.

**Kosten und Risiken:**

- In Teams, die es gewohnt sind, unter Druck mit minimalem Prozess auszuliefern, wird die Einführung einer DoD oft als Bürokratie wahrgenommen, besonders wenn die Kriterien innerhalb der bestehenden Sprint-Kapazität nicht erfüllt werden können — dies erzeugt Druck, den Standard aufzuheben, was schlimmer ist als keinen zu haben.
- Legacy-Systemen fehlt oft die Testinfrastruktur, Deployment-Automatisierung und Dokumentations-Tooling, die nötig sind, um eine sinnvolle DoD zu erfüllen; die DoD kann Lücken in der Engineering-Plattform aufdecken, die separate Investition erfordern.
- Eine übermäßig ambitionierte DoD in einem Legacy-Kontext kann den Fortschritt stoppen, wenn Teams das Gefühl haben, dass angesichts des Zustands des umgebenden Systems fast nichts „fertig" sein kann — die DoD muss auf das kalibriert werden, was jetzt erreichbar ist, nicht auf das, was ideal wäre.
- Ohne Managementunterstützung für die Zeit, die nötig ist, um DoD-Kriterien zu erfüllen, erleben Teams die DoD als zusätzliches unfinanziertes Mandat auf einem bereits anspruchsvollen Lieferplan.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Teams eine Definition of Done in Legacy-Umgebungen einführten und welche Auswirkung sie hatte.

Das ERP-Anpassungsteam eines Fertigungsunternehmens hatte Jahre damit verbracht, einem SAP-System Features ohne konsistentes Qualitätstor hinzuzufügen. Entwickler betrachteten eine Änderung als fertig, wenn der Fachanwender bestätigte, dass sie in seinem manuellen Test funktionierte. Nach zwei kostspieligen Rollbacks, verursacht durch Änderungen, die nicht gegen die nächtlichen Batch-Jobs getestet worden waren, führte das Team eine DoD ein, die einen verpflichtenden Batch-Job-Simulationsschritt in der Staging-Umgebung enthielt, bevor eine Änderung abgeschlossen werden konnte. Dieses eine Kriterium eliminierte das Rollback-Muster innerhalb eines Quartals, weil es das Team zwang, die Testinfrastruktur zu bauen, die sie jahrelang aufgeschoben hatten.

Das Patientenaktensystem eines Gesundheitsanbieters hatte über fünfzehn Jahre Ad-hoc-Anfragen Hunderte undokumentierter Felder angehäuft. Als ein neues Team die Modernisierung übernahm, führten sie ein DoD-Kriterium ein, das verlangte, dass jede Änderung am Datenmodell eine kurze Beschreibung des Feldzwecks in einem gemeinsamen Datenwörterbuch enthält. Das Team versuchte nicht, alle bestehenden Felder auf einmal zu dokumentieren; stattdessen stellte die DoD sicher, dass jedes während normaler Entwicklung berührte Feld als Nebeneffekt der bereits geleisteten Arbeit dokumentiert wurde. Innerhalb von zwei Jahren waren die am häufigsten aufgerufenen Teile des Datenmodells vollständig dokumentiert, ohne ein dediziertes Dokumentationsprojekt.

Eine Einzelhandelsbank, die ein Kreditvergabesystem betrieb, hatte ein anhaltendes Problem mit Features, die QA bestanden, aber in Produktion wegen Konfigurationsunterschieden zwischen Umgebungen versagten. Das Team führte ein DoD-Kriterium ein, das verlangte, dass alle neuen Konfigurationsparameter zu einer Umgebungskonfigurations-Checkliste hinzugefügt und in der produktionsäquivalenten Staging-Umgebung verifiziert werden, bevor eine Story geschlossen wird. Dieses Kriterium erzwang die Schaffung eines Konfigurationsmanagementprozesses, der nie existiert hatte, und eliminierte eine ganze Kategorie von Produktionsvorfällen, die zuvor Notfall-Patches erfordert hatten.
