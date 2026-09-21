---
title: Verständliche Fehlermeldungen
description: Anzeige klarer, kontextbezogener Fehlermeldungen, wenn
  Probleme auftreten.
category:
- Requirements
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/understandable-error-messages/
problems:
- user-confusion
- user-frustration
- poor-user-experience-ux-design
- inadequate-error-handling
- negative-user-feedback
- increased-customer-support-load
- increased-error-rates
- user-trust-erosion
- negative-brand-perception
layout: solution
lang: de
en_slug: understandable-error-messages
related_solutions:
- slug: plain-language
  similarity: 0.8
- slug: confirmation-dialogs
  similarity: 0.75
- slug: consistent-terminology
  similarity: 0.75
- slug: contextual-help
  similarity: 0.75
- slug: error-reporting-and-analysis
  similarity: 0.7
- slug: intuitive-navigation
  similarity: 0.7
---

## Description

Eine verständliche Fehlermeldung erklärt in einfacher Sprache, was passiert ist, warum, und was der Nutzer dagegen tun kann, statt die rohe technische Exception, den Datenbank-Constraint-Code oder den Stack Trace offenzulegen, den Legacy-Systeme routinemäßig direkt aus der Schicht anzeigen, in der der Fehlschlag tatsächlich auftrat. Eine Meldung wie eine rohe Oracle-Constraint-Verletzung bedeutet der Person, die sie auslöste, nichts und wird zuverlässig zu einem Support-Anruf, während derselbe Fehlschlag, übersetzt in das, was er für ihre spezifische Aktion bedeutet, es ihnen erlaubt, ihn selbst zu beheben. Den vollständigen Fehlermeldungskatalog eines Legacy-Systems auf diese Weise neu zu schreiben ist genuin substanzielle Arbeit über Hunderte von Aufrufstellen hinweg, und es hat auch eine echte Sicherheitsdimension — das technische Detail muss weiterhin serverseitig zur Diagnose protokolliert werden, nur nie dem Endnutzer offengelegt werden, wo es sonst interne Implementierungsdetails neben der Verwirrung durchsickern lassen würde.

## How to Apply ◆

> Legacy-Systeme zeigen oft rohe technische Fehlermeldungen, Stack Traces oder kryptische Fehlercodes an, die für Endnutzer bedeutungslos sind. Verständliche Fehlermeldungen erklären Probleme in menschlichen Begriffen und leiten die Wiederherstellung.

- Prüfen Sie alle Fehlermeldungen im Legacy-System und kategorisieren Sie sie nach Schweregrad, Häufigkeit und Nutzerauswirkung. Priorisieren Sie die Neuformulierung der häufigsten und verwirrendsten Meldungen zuerst.
- Strukturieren Sie jede Fehlermeldung mit drei Komponenten: was passiert ist, warum es passiert ist (wenn bekannt), und was der Nutzer dagegen tun kann. Dieses Muster stellt sicher, dass jede Meldung umsetzbar ist.
- Ersetzen Sie technische Bezeichner und Codes durch einfache Sprache. Wenn Fehlercodes zu Support-Zwecken beibehalten werden müssen, zeigen Sie sie in sekundärer Position an, wie "Fehlercode: 4021", angehängt an die menschenlesbare Meldung.
- Positionieren Sie Fehlermeldungen neben dem Element, das den Fehler verursachte, statt in einem generischen Benachrichtigungsbereich. Heben Sie für Formularvalidierung das spezifische Feld hervor und platzieren Sie die Meldung daneben.
- Differenzieren Sie Fehlerschweregrad visuell: nutzen Sie unterschiedliche Gestaltung für Validierungswarnungen, nutzerbehebbare Fehler und Systemfehler, die Administratoreingriff erfordern.
- Protokollieren Sie die technischen Details von Fehlern serverseitig zum Debuggen, während Sie nur nutzerrelevante Informationen in der UI zeigen. Nutzer sollten nie Stack Traces, SQL-Fehler oder interne Exception-Meldungen sehen.

## Tradeoffs ⇄

> Klare Fehlermeldungen verwandeln frustrierende Sackgassen in behebbare Situationen, erfordern aber Investition in die Neuformulierung von Meldungen und deren Pflege.

**Vorteile:**

- Ermöglicht Nutzern, Fehler unabhängig zu lösen, statt Support zu kontaktieren, was das Support-Ticket-Volumen direkt reduziert.
- Reduziert Nutzerfrustration und Vertrauenserosion, verursacht durch die Konfrontation mit unverständlichen technischen Meldungen.
- Senkt Fehlerraten, indem Nutzern geholfen wird zu verstehen, was schiefgelaufen ist und wie sie ihre Eingabe oder ihren Ansatz korrigieren können.
- Beseitigt das Sicherheitsrisiko, interne Systemdetails durch technische Fehlermeldungen offenzulegen.

**Kosten und Risiken:**

- Die Neuformulierung Hunderter Fehlermeldungen in einem großen Legacy-System erfordert Zeit und Zusammenarbeit zwischen Entwicklern, die die Fehler verstehen, und Redakteuren, die klar kommunizieren können.
- Übermäßig vereinfachte Fehlermeldungen, die relevante Details verbergen, können es Support-Personal erschweren, Probleme zu diagnostizieren, wenn Nutzer sie tatsächlich kontaktieren.
- Fehlermeldungen müssen aktualisiert werden, wenn sich das Systemverhalten ändert, sonst liefern sie falsche Anleitung.
- Die Internationalisierung von Fehlermeldungen fügt Übersetzungsaufwand für jede unterstützte Sprache hinzu.

## How It Could Be

> Kryptische Fehlermeldungen gehören zu den universell frustrierendsten Aspekten von Legacy-Systemen und einem der am leichtesten inkrementell zu verbessernden.

Ein Legacy-Bestandsverwaltungssystem zeigt rohe Datenbank-Constraint-Verletzungsmeldungen an, wenn Nutzer Operationen versuchen, die Geschäftsregeln widersprechen. Eine Meldung wie "ORA-02292: integrity constraint (INV.FK_ITEM_WAREHOUSE) violated - child record found" erscheint, wenn ein Lagerleiter versucht, ein Lager zu deaktivieren, das noch Bestand enthält. Der Leiter hat keine Ahnung, was die Meldung bedeutet, und ruft den IT-Support an. Das Team ordnet die fünfzig häufigsten Datenbankfehler nutzerfreundlichen Meldungen zu. Die Constraint-Verletzungsmeldung wird zu "Dieses Lager kann nicht deaktiviert werden, weil es noch aktive Bestandsartikel enthält. Bitte übertragen oder buchen Sie alle Artikel aus, bevor Sie deaktivieren." und beinhaltet einen Link zum Bestandsübertragungsbildschirm. Support-Anrufe im Zusammenhang mit Fehlermeldungen sinken dramatisch, und Lagerleiter berichten, sich zuversichtlicher bei der Verwaltung von Lagerkonfigurationen zu fühlen, weil sie die Einschränkungen des Systems verstehen.
