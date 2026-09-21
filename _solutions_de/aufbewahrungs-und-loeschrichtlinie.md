---
title: Aufbewahrungs- und Löschrichtlinie
description: Festlegung, was in welcher Form wie lange aufbewahrt werden
  muss und was gelöscht werden muss, sodass die Aufbewahrungspflicht die
  Daten einschränkt, statt das haltende System einzufrieren.
category:
- Security
- Operations
- Management
problems:
- retention-obligations-block-change
- regulatory-compliance-drift
- obsolete-technologies
- high-maintenance-costs
- modernization-strategy-paralysis
- data-migration-complexities
- lack-of-ownership-and-accountability
- vendor-dependency-entrapment
- legal-disputes
layout: solution
lang: de
en_slug: retention-and-disposal-policy
related_solutions:
- slug: system-decommissioning
  similarity: 0.75
- slug: datensparsamkeit
  similarity: 0.7
- slug: backup-and-recovery
  similarity: 0.65
- slug: feature-usage-measurement
  similarity: 0.6
- slug: role-model-rationalization
  similarity: 0.6
- slug: technology-radar
  similarity: 0.6
---

## Description

Eine Aufbewahrungs- und Löschrichtlinie legt pro Datenkategorie fest, was die Organisation behalten muss, wie lange, in welcher Form es abrufbar bleiben muss, und — die Hälfte, die üblicherweise fehlt — was gelöscht werden muss, sobald die Frist endet. Ihr Zweck in einem Legacy-Kontext ist es, die Verpflichtung vom System zu trennen. Organisationen schließen routinemäßig, dass ein System aufgrund von Aufbewahrungspflichten nicht stillgelegt werden kann, ohne je festzulegen, an welche Daten diese Pflichten geknüpft sind oder was „abrufbar" tatsächlich erfordert. Diese Vermischung ist es, was eine Datenverpflichtung in ein permanent finanziertes System verwandelt. Eine Richtlinie, die Verpflichtungen spezifischen Artefakten zuordnet, macht die Alternative sichtbar: die Artefakte mit einer nachweisbaren Integritätsgarantie bewahren, und das System, das sie produziert hat, wird stilllegbar wie jedes andere.

## How to Apply ◆

> Der Grund, warum ein System nicht außer Betrieb genommen werden kann, ist fast nie die Aufbewahrungspflicht selbst; es ist, dass niemand festgelegt hat, was die Pflicht tatsächlich abdeckt.

- **Ordnen Sie Verpflichtungen Datenkategorien zu, nicht Systemen.** Zeichnen Sie für jede Kategorie die Quelle der Verpflichtung, die Frist, wann die Uhr zu laufen beginnt, und in welcher Form die Daten bleiben müssen. Eine Richtlinie, die ihre Quelle nicht zitieren kann, wird ihre erste Anfechtung nicht überleben.
- **Tun Sie dies gemeinsam zwischen Recht und Technologie.** Recht weiß, was die Verpflichtung besagt; Technologie weiß, was die Daten sind und was ihre Bewahrung erfordern würde. Keiner allein kann eine nutzbare Richtlinie produzieren, und jeder hat historisch angenommen, es sei das Thema des anderen.
- **Legen Sie fest, was „abrufbar" und „lesbar" in der Praxis erfordern.** Einen Datensatz in fünfzehn Jahren zu produzieren bedeutet im Allgemeinen, seinen Inhalt, seinen Kontext und genug seiner Bedeutung zu bewahren, um interpretierbar zu sein — nicht die Anwendung zu bewahren, die ihn anzeigte. Dies explizit festzustellen ist es, was Stilllegung freischaltet.
- **Definieren Sie die Löschseite mit gleicher Präzision.** Aufbewahrung hat eine Obergrenze ebenso wie eine Untergrenze, und in vielen Rechtsordnungen ist das Behalten personenbezogener Daten über ihre Frist hinaus selbst ein Verstoß. Richtlinien, die nur das Behalten adressieren, sind halb geschrieben und schaffen Exposition statt sie zu beseitigen.
- **Weisen Sie jeder Datenkategorie einen Eigentümer zu**, verantwortlich dafür, dass die Frist korrekt ist und dass Löschung tatsächlich geschieht. Aufbewahrung ohne Eigentümer produziert standardmäßig unbegrenzte Anhäufung.
- **Bevorzugen Sie ein Archiv mit Integritätsgarantien gegenüber einem laufenden System.** Bewahrte Artefakte mit Prüfsummen, Zeitstempeln und einem Audit-Trail erfüllen die meisten Verpflichtungen zu einem Bruchteil der Kosten, die betreffende Anwendung am Leben zu halten.
- **Testen Sie Abruf, wiederholt.** Ein Archiv, aus dem niemand gelesen hat, ist eine Hoffnung. Periodische Übungen, einen Datensatz aus der ältesten aufbewahrten Periode abzurufen, sind es, die feststellen, ob die Regelung funktioniert.
- **Behandeln Sie Legal Hold als separaten Mechanismus**, der Löschung für identifizierte Datensätze aussetzt. Ohne ihn löscht eine Organisation entweder etwas unter Hold oder setzt sicherheitshalber alle Löschung unbegrenzt aus.
- **Überprüfen Sie die Richtlinie gegen sich ändernde Verpflichtungen** in einem festen Rhythmus. Fristen und Anforderungen ändern sich, und eine einmal festgelegte, nie überprüfte Richtlinie driftet in beide Richtungen aus der Compliance.

## Tradeoffs ⇄

> Eine präzise Richtlinie verwandelt eine vage Einfrierung in eine begrenzte Verpflichtung, erfordert aber juristisches Urteilsvermögen, und eine falsch gesetzte Frist oder Form trägt Konsequenzen, die eher rechtlicher als betrieblicher Natur sind.

**Vorteile:**

- Systeme, die allein als Datenverwahrer am Leben gehalten werden, werden stilllegbar, was häufig eine große und permanente Kostenreduktion ist.
- Die Verpflichtung wird begrenzt und spezifisch, sodass Modernisierungsoptionen dagegen bewertet werden können, statt daran zu scheitern.
- Löschung geschieht tatsächlich, was sowohl die Speicherkosten als auch die Exposition beseitigt, die aus dem Behalten personenbezogener Daten über ihre rechtmäßige Frist hinaus entsteht.
- Migrationen werden machbar, weil das, was bewahrt werden muss, festgelegt und nachweisbar ist statt als alles angenommen zu werden.
- Audit- und regulatorische Anfragen können aus einer Richtlinie und einem Archiv beantwortet werden statt aus einer Archäologie-Übung.

**Kosten und Risiken:**

- Die Bestimmung der geltenden Verpflichtungen erfordert juristisches Fachwissen, und in multinationalen Organisationen ist die Analyse echt komplex.
- Eine falsch gesetzte Frist oder Form produziert eine rechtliche Exposition, die möglicherweise erst nach Jahren auftaucht, zu welchem Zeitpunkt sie nicht wiederherstellbar ist.
- Die Migration aufbewahrter Daten in ein Archiv muss Bedeutung bewahren, und das zu demonstrieren ist schwerer, als die Datensätze zu verschieben.
- Löschung ist irreversibel, sodass ein Fehler in der Richtlinie Daten zerstört, die sich als benötigt herausstellen.
- Die Arbeit ist unglamourös und liefert keine Fähigkeit, sodass sie schwer gegen alles zu finanzieren ist, was es tut.

## How It Could Be

Ein Versicherer hielt drei abgelöste Systeme allein am Laufen, um Aufbewahrungspflichten zu erfüllen, die sich über dreißig Jahre erstreckten. Ein gemeinsames Recht-und-Technologie-Review bildete zum ersten Mal die Verpflichtungen ab und fand, dass sie an spezifische Artefakte geknüpft waren — das Policendokument, eine definierte Menge von Transaktionsaufzeichnungen und Korrespondenz — statt an das Betriebssystem, das sie produziert hatte. Die Bewahrung dieser Artefakte in einem Archiv mit Integritätsgarantien und einem getesteten Abrufverfahren erfüllte die Anforderung. Zwei der drei Systeme wurden innerhalb eines Jahres außer Betrieb genommen, was Lizenz-, Infrastruktur- und Spezialauftragnehmerkosten beendete, die neun Jahre lang jährlich erneuert worden waren, ohne dass jemand untersucht hatte, was sie kauften.

Die Löschseite produzierte den unbequemeren Befund. Ungefähr 40 Prozent der aufbewahrten Daten waren über jede geltende Frist hinaus, einschließlich personenbezogener Daten, deren fortgesetzte Aufbewahrung an sich schon ein Verstoß war. Die Organisation hatte ein Jahrzehnt lang unter der Annahme operiert, dass Aufbewahrung Behalten bedeutete, und hatte Löschung überhaupt nie implementiert — es gab keinen Prozess, keinen Eigentümer und keinen Mechanismus. Einen zu etablieren dauerte länger als das Archiv, größtenteils weil es erforderte, dass jemand die Verantwortung für die Löschung von Datensätzen übernahm, was sich als eine Entscheidung herausstellte, die niemand je gebeten worden war zu treffen.
