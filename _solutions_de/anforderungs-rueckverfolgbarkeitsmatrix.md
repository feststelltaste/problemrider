---
title: Anforderungs-Rückverfolgbarkeitsmatrix
description: Pflege expliziter bidirektionaler Zuordnungen von Anforderungen
  über Design, Code bis zu Tests.
category:
- Requirements
- Process
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- misaligned-deliverables
- insufficient-testing
- poor-test-coverage
- regulatory-compliance-drift
- legacy-system-documentation-archaeology
- feature-gaps
- legal-disputes
- poor-contract-design
layout: solution
lang: de
en_slug: requirements-traceability-matrix
related_solutions:
- slug: requirements-analysis
  similarity: 0.75
- slug: evolutionary-requirements-development
  similarity: 0.7
- slug: compatibility-matrix
  similarity: 0.7
- slug: story-mapping
  similarity: 0.7
- slug: user-stories
  similarity: 0.65
- slug: architecture-decision-records
  similarity: 0.65
---

## Description

Eine Anforderungs-Rückverfolgbarkeitsmatrix ist eine explizite, bidirektionale Zuordnung, die jede Geschäftsanforderung mit dem Code, den Datenbankstrukturen und den Tests verknüpft, die sie implementieren oder verifizieren, und macht eine Beziehung sichtbar, die in den meisten Legacy-Systemen nur implizit existiert, wenn überhaupt. Eine solche zu bauen bedeutet typischerweise, das tatsächliche Verhalten des Systems und jede überlebende Dokumentation zu reverse-engineeren, um zu rekonstruieren, welche Anforderungen der Code ursprünglich erfüllen sollte, da die ursprünglichen Anforderungsdokumente — falls sie je existierten — üblicherweise verloren gegangen oder ersetzt wurden, lange bevor die aktuellen Betreuer ankamen. Dies zählt akut in der Legacy-Modernisierung, weil ohne eine solche Zuordnung jede vorgeschlagene Änderung oder Migration verstecktes Risiko trägt: Ein Modul, das wie toter Code aussieht, könnte tatsächlich die einzige Implementierung einer regulatorischen Anforderung sein, und eine Anforderung, die erfüllt aussieht, könnte in Wirklichkeit keinen automatisierten Test haben, der sie schützt. Die Matrix verwandelt dieses unsichtbare Risiko in eine sichtbare Arbeitsliste, die genau zeigt, welchen Anforderungen Testabdeckung fehlt, welcher Code keiner aktiven Anforderung mehr entspricht und daher Kandidat für Entfernung ist, und welche Teile des Systems verifiziert werden müssen, bevor eine Legacy-Komponente sicher außer Betrieb genommen werden kann. Sie ist besonders wertvoll in regulierten Branchen, wo Auditoren dokumentierte Evidenz erwarten, dass jede compliance-relevante Anforderung sowohl implementiert als auch getestet ist, Evidenz, die das Stammeswissen eines Legacy-Systems allein nicht liefern kann. Da die Matrix in dem Moment, in dem sie nicht mehr aktualisiert wird, zu aktiv irreführender Dokumentation verfällt, hängt ihr Wert vollständig davon ab, ihre Pflege als ständigen Teil des Änderungsprozesses zu behandeln statt als einmalige Rekonstruktionsübung.

## How to Apply ◆

> In Legacy-Systemen hilft eine Anforderungs-Rückverfolgbarkeitsmatrix Teams zu verstehen, welche Teile der Codebasis welche Geschäftsanforderungen implementieren — Wissen, das über Jahre undokumentierter Änderungen oft vollständig verloren geht.

- Beginnen Sie mit der Inventarisierung der bekannten Geschäftsanforderungen, die das Legacy-System erfüllt, unter Nutzung jeder verfügbaren Dokumentation, Nutzerinterviews und Analyse der bestehenden Codebasis.
- Erstellen Sie eine Matrix, die jede Anforderung den Codemodulen, Datenbankobjekten und Tests zuordnet, die sie implementieren oder verifizieren, selbst wenn die Zuordnung anfangs unvollständig ist.
- Nutzen Sie die Matrix, um ungetestete Anforderungen zu identifizieren — dies sind Hochrisikobereiche, in denen Änderungen kritische Funktionalität ohne automatisierte Erkennung brechen könnten.
- Nutzen Sie bei der Planung von Modernisierungsarbeit die Matrix, um die vollständige Auswirkung des Ersetzens oder Modifizierens einer spezifischen Geschäftsfähigkeit zu bestimmen.
- Aktualisieren Sie die Matrix als Teil jeder Änderung am System und machen Sie Rückverfolgbarkeitspflege zu einer Standardpraxis statt zu einer einmaligen Dokumentationsübung.
- Nutzen Sie die Matrix während Compliance-Audits, um zu demonstrieren, dass regulatorische Anforderungen implementiert und verifiziert sind, was besonders wichtig in regulierten Branchen ist, die Legacy-Systeme modernisieren.

## Tradeoffs ⇄

> Eine Rückverfolgbarkeitsmatrix bietet unschätzbare Sichtbarkeit in Legacy-Systeme, erfordert aber anhaltenden Aufwand zur Erstellung und Pflege.

**Vorteile:**

- Macht die Beziehung zwischen Geschäftsanforderungen und Implementierung explizit und reduziert das Risiko, kritische Funktionalität während der Modernisierung versehentlich zu entfernen oder zu brechen.
- Ermöglicht Impact-Analyse für vorgeschlagene Änderungen, indem genau gezeigt wird, welche Anforderungen, welcher Code und welche Tests betroffen sind.
- Unterstützt Compliance- und Audit-Anforderungen, indem dokumentierte Evidenz geliefert wird, dass regulatorische Anforderungen implementiert und getestet sind.
- Hilft, verwaisten Code zu identifizieren — Implementierung, die keiner aktiven Anforderung mehr entspricht und potenziell entfernt werden kann.

**Kosten und Risiken:**

- Der Bau der anfänglichen Matrix für ein Legacy-System mit schlechter Dokumentation ist ein erheblicher Aufwand, der Wochen an Reverse Engineering erfordern kann.
- Wenn die Matrix nicht gepflegt wird, während sich das System weiterentwickelt, wird sie irreführend — schlimmer, als gar keine Matrix zu haben.
- Übermäßig detaillierte Matrizen erzeugen Pflegeaufwand, den Teams unter Lieferdruck möglicherweise aufgeben.
- Die Matrix ist nur so gut wie das Verständnis des Teams von den Anforderungen des Legacy-Systems, das selbst unvollständig oder falsch sein könnte.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie eine Rückverfolgbarkeitsmatrix Legacy-Modernisierung in einer regulierten Umgebung unterstützt.

Ein Pharmaunternehmen modernisierte sein Laborinformationsmanagementsystem (LIMS), das 18 Jahre im Einsatz gewesen war. Regulatorische Anforderungen schrieben vor, dass jede Berechnung im System zu einer validierten Anforderung rückverfolgbar und durch einen dokumentierten Test abgedeckt sein muss. Das Team baute eine Rückverfolgbarkeitsmatrix durch Reverse-Engineering der Legacy-Codebasis und ordnete 340 regulatorische Anforderungen spezifischen Codemodulen und bestehenden Testfällen zu. Die Matrix offenbarte, dass 45 Anforderungen keine entsprechenden Tests hatten und 23 Tests hatten, die nicht mehr bestanden. Diese Analyse trieb den Testbehebungsplan an und gab Regulierungsbehörden Vertrauen, dass die Modernisierung Compliance aufrechterhalten würde. Während der Migration diente die Matrix als Checkliste — jede Anforderung wurde einzeln im neuen System verifiziert, bevor das entsprechende Legacy-Modul außer Betrieb genommen wurde.
