---
title: Wissensmanagement-System
description: Zentrale Sammlung und Verteilung von Wissen über das Softwareprojekt.
category:
- Communication
- Team
quality_tactics_url: https://qualitytactics.de/en/maintainability/knowledge-management-system/
problems:
- knowledge-silos
- tacit-knowledge
- implicit-knowledge
- knowledge-gaps
- knowledge-sharing-breakdown
- difficult-developer-onboarding
- information-decay
- legacy-system-documentation-archaeology
- slow-knowledge-transfer
- team-silos
- duplicated-research-effort
- duplicated-effort
- extended-research-time
- technology-isolation
- incomplete-knowledge
- inconsistent-knowledge-acquisition
- feedback-isolation
- knowledge-dependency
- communication-risk-within-project
layout: solution
lang: de
en_slug: knowledge-sharing-practices
related_solutions:
- slug: knowledge-base
  similarity: 0.85
- slug: knowledge-rotation
  similarity: 0.8
- slug: documentation-as-code
  similarity: 0.8
- slug: architecture-decision-records
  similarity: 0.8
- slug: runbooks
  similarity: 0.75
- slug: pair-and-mob-programming
  similarity: 0.75
---

## Description

Ein Wissensmanagement-System erfasst bewusst institutionelles Wissen — Architekturentscheidungen, Betriebs-Runbooks, undokumentierte Geschäftsregeln, die beim Debuggen entdeckt werden —, sobald es auftaucht, statt Dokumentation als ein Projekt zu behandeln, das später angegangen wird. In Legacy-Systemen ist dies weniger eine Dokumentations-Nettigkeit als eine direkte Risikominderungsstrategie, da so viel von dem, was das System am Laufen hält, nur in den Köpfen einer kleinen Zahl von Menschen existiert, deren Weggang eine Routinefrage in eine mehrtägige Untersuchung oder einen Produktionsvorfall verwandeln kann. Mit den Komponenten zu beginnen, die den längsten Ausfall verursachen würden, wenn die eine kundige Person morgen ginge, und eine „Drei-Treffer"-Regel zu übernehmen, die jede dritte gestellte Frage zu einem permanenten Artikel macht, wandelt verstreutes Erfahrungswissen in etwas Durchsuchbares um — obwohl das System nur nützlich bleibt, wenn es überprüft und aktualisiert wird, während sich das System weiterentwickelt, da veraltete Legacy-Dokumentation jemanden aktiv dazu verleiten kann, das falsche Verfahren an einem fragilen System auszuführen.

## How to Apply ◆

> In Legacy-System-Kontexten ist ein Wissensmanagement-System nicht primär ein Dokumentationswerkzeug — es ist eine Risikominderungsstrategie für das institutionelle Wissen, das nur in den Köpfen von Menschen existiert, die gehen könnten.

- Beginnen Sie damit, die kritischsten Wissenslücken des Teams zu identifizieren: Welche Teile des Legacy-Systems würden den längsten Ausfall verursachen, wenn die eine Person, die sie versteht, morgen ginge? Dokumentieren Sie diese zuerst, vor allem anderen.
- Etablieren Sie Architecture Decision Records (ADRs) für Entscheidungen, die im Legacy-System bereits getroffen wurden, rückwärts arbeitend von Code, der seltsam oder überraschend aussieht, um die ursprüngliche Begründung aufzudecken — dies wandelt stilles Erfahrungswissen in durchsuchbares institutionelles Gedächtnis um.
- Erstellen Sie Runbooks für jedes wiederkehrende Betriebsverfahren, das derzeit im Muskelgedächtnis von jemandem lebt: Deployments, Batch-Neustarts, Datenbank-Failover, Monatsend-Verarbeitungssequenzen und alle manuellen Schritte, die automatisierte Prozesse begleiten.
- Erfassen Sie entdecktes Legacy-Verhalten — undokumentierte API-Verträge, implizite Geschäftsregeln, vergraben in gespeicherten Prozeduren, umgebungsspezifische Eigenheiten —, sobald es während Wartung und Debugging auftaucht, nicht später; „später" kommt in Legacy-Arbeit selten.
- Übernehmen Sie eine „Drei-Treffer"-Regel: Wenn eine Frage über Legacy-Verhalten zum dritten Mal über irgendeinen Kanal gestellt wird (Chat, E-Mail, Flurgespräch), wird die Antwort zu einem Wissensdatenbankartikel, bevor das Gespräch endet.
- Strukturieren Sie Onboarding-Material um die spezifischen Herausforderungen des Legacy-Systems herum — welche Module am gefährlichsten anzufassen sind, welche Datenbanktabellen gemeinsam genutzt werden und warum, welche Deployment-Schritte manuell sind — statt generischer Softwareentwicklungsleitfäden.
- Machen Sie das Wissenssystem von innerhalb der Werkzeuge auffindbar, die Entwickler tatsächlich nutzen: Verlinken Sie relevante Runbooks von Monitoring-Alarmen, betten Sie ADR-Referenzen in Codekommentare nahe den Entscheidungen ein, die sie erklären, und verlinken Sie Fehlerbehebungsleitfäden von CI-Fehlermeldungen.
- Weisen Sie die Eigentümerschaft von Wissensabschnitten bestimmten Teams oder Einzelpersonen zu und setzen Sie einen Review-Takt; unüberprüfte Legacy-Dokumentation verkommt schnell, während sich das System weiterentwickelt, und wird aktiv irreführend.

## Tradeoffs ⇄

> Ein Wissensmanagement-System erfordert anhaltende Investition in eine Disziplin — Dinge aufschreiben —, zu der Entwicklungsteams strukturell fehlanreiziert sind, aber in Legacy-Kontexten wird die Kosten des Nicht-Investierens in Ausfallstunden und gescheiterten Modernisierungsversuchen gemessen.

**Vorteile:**

- Schützt die Organisation vor der gefährlichsten Form von Legacy-Risiko: dem Weggang einer Schlüsselperson, die kritisches Systemwissen vollständig im Kopf trägt.
- Beschleunigt die Einarbeitung neuer Entwickler in ein Legacy-System, indem strukturierte Erklärungen nicht offensichtlichen Verhaltens bereitgestellt werden, was die Zeit von der Einstellung bis zum produktiven Beitrag verkürzt.
- Verringert die Dauer von Produktionsvorfällen, indem Fehlerbehebungsleitfäden durchsuchbar und für Entwickler verfügbar gemacht werden, die die betroffene Komponente nicht ursprünglich gebaut haben.
- Schafft eine Prüfspur dafür, warum Legacy-Designentscheidungen getroffen wurden, was den wiederholten Zyklus verhindert, in dem ein neues Teammitglied eine Änderung vorschlägt, hört „das haben wir schon versucht" und nicht herausfinden kann, warum es fehlschlug.
- Unterstützt Modernisierungsplanung, indem der Umfang und die Struktur des Legacy-Systems für Architekten und Stakeholder lesbar gemacht werden, die nicht am Bau beteiligt waren.

**Kosten und Risiken:**

- Der anfängliche Aufwand, ein Legacy-System zu dokumentieren — besonders eines mit Jahren undokumentierten Verhaltens — ist beträchtlich und konkurriert direkt mit Lieferarbeit, die für Stakeholder unmittelbarer sichtbar ist.
- Legacy-Wissen verkommt schnell, wenn das System aktiv gewartet wird; vor sechs Monaten geschriebene Dokumentation könnte bereits falsch sein, und veraltete Dokumentation in einem Legacy-Kontext ist besonders gefährlich, weil sie Entwickler dazu bringen kann, falsche Verfahren an fragilen Systemen auszuführen.
- Ohne eine Kultur, die Wissensaustausch wertschätzt und belohnt, wird das System zu einem Nur-Schreib-Archiv: Erfahrene Entwickler tragen nicht bei, und neue Entwickler lernen, ihm nicht zu vertrauen.
- Legacy-Systeme enthalten oft Wissen, das politisch sensibel ist — Entscheidungen, getroffen aus Gründen, die organisatorisches Versagen widerspiegeln, Workarounds für Managemententscheidungen, die nicht infrage gestellt werden können —, und dieses Wissen wird häufig aus der Dokumentation ausgelassen, selbst wenn es kritisch fürs Systemverständnis ist.
- Die Wahl von Tooling, das nicht mit dem bestehenden Workflow des Teams integriert, führt zu einem Wissenssystem, das während Dokumentations-Sprints gepflegt und den Rest der Zeit ignoriert wird.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Wissensmanagement-Praktiken die Fragilität von Legacy-System-Betrieb verringert haben.

Eine Regierungsbehörde betrieb ein Ende der 1980er Jahre gebautes Rentenberechnungssystem. Der Hauptpfleger des Systems ging in Rente, und innerhalb von drei Monaten hatte das verbleibende Team zwei falsche Leistungsberechnungen erlebt, verursacht durch Randfälle, die nur der Rentner verstanden hatte. Die Behörde reagierte, indem sie den ehemaligen Pfleger für sechs Monate als Teilzeitberater mit dem expliziten Ziel der Wissensexternalisierung einstellte. In Zusammenarbeit mit zwei Junior-Entwicklern dokumentierte der Berater die Berechnungslogik für dreiundvierzig Leistungsszenarien, die historischen Gründe für fünfzehn sonst unerklärliche Code-Entscheidungen und das manuelle Korrekturverfahren für vier vierteljährliche Abgleichsschritte, die nie automatisiert worden waren. Die resultierende Wissensdatenbank verringerte Produktionsvorfälle mit der Berechnungs-Engine von durchschnittlich acht pro Jahr auf einen in den folgenden zwei Jahren.

Die E-Commerce-Plattform eines Einzelhandelsunternehmens war elf Jahre lang von einem einzigen Offshore-Anbieter gepflegt worden. Als der Vertrag endete und die Arbeit ins Haus geholt wurde, entdeckte das eingehende Team, dass praktisch keine Dokumentation existierte. Der Wissenstransfer bestand aus zwei Wochen Shadowing-Sitzungen und einem Satz exportierter Chat-Protokolle. Das neue Team etablierte vom ersten Tag an eine Wissensdatenbank und verlangte, dass jede Bug-Untersuchung mit einem Wissensartikel endete, der Symptom, Grundursache und Behebung beschrieb. Innerhalb von vier Monaten hatten sie 180 Artikel angesammelt, die die häufigsten Fehlermodi des Systems abdeckten. Neue Teammitglieder wurden mittels einer strukturierten Leseliste aus diesen Artikeln eingearbeitet, was die Zeit bis zur ersten unbegleiteten Fehlerbehebung von sechs Wochen auf zwei verkürzte.

Eine Bank, die ein COBOL-basiertes Zahlungsabwicklungssystem betrieb, stellte fest, dass ihre erfahrensten Mainframe-Entwickler sich gleichzeitig der Rente näherten. Statt auf die Weggänge zu warten, etablierte die IT-Führung ein verpflichtendes Wissenserfassungsprogramm: Jeder leitende Entwickler verbrachte vier Stunden pro Woche damit, mit einem Junior-Entwickler zu paaren, um die Abwicklungslogik durchzugehen, die JCL-Job-Streams zu dokumentieren und die Ausnahmebehandlungsverfahren zu erfassen, die nur in den täglichen Routinen der leitenden Entwickler existierten. Die Sitzungen wurden mittels einer Vorlage strukturiert, die den geschäftlichen Zweck, den technischen Mechanismus und die Fehlermodi für jede Komponente erfasste. Über achtzehn Monate baute die Bank einen Korpus von 340 dokumentierten Komponenten auf, der 90 % des täglichen Abwicklungsverarbeitungsvolumens abdeckte, was die Schlüsselpersonenabhängigkeit der Organisation vor Beginn der Renteneintritte substanziell verringerte.
